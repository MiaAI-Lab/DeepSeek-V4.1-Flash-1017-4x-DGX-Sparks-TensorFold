#!/usr/bin/env bash
# Serve DeepSeek-V4.1-Flash-1017 (deepseek-ai's own FP8 / MXFP4 release) on four DGX Sparks on a switch with TensorFold's
# native engine (four Sparks cabled as a ring without a switch: ./start-switchless.sh, the same script), end to end: runs scripts/prepare.sh when the image, the checkpoint, the weight files, the kit or the workers'
# NFS volumes are not ready yet (the first run, or after the patches change), starts ranks 3, 2 and 1 on the workers and
# rank 0 here, which serves the API on port 8888, waits until the OpenAI API answers, then runs a smoke test. Every
# Spark reads the weights from this one over NFS. Stop it with ./stop.sh.
#
# Usage: ./start.sh [restart] [extra tensorfold-native serve args]
#   ./start.sh                  # scripts/config.sh's defaults (16 requests at once, a 1,048,576-token window, drafts,
#                               # images); if the server already runs, says so and leaves it alone
#   ./start.sh restart          # stop every rank (./stop.sh), then start again (e.g. after changing settings)
#   CONTEXT=262144 ./start.sh restart
#   DRY_RUN=1 ./start.sh        # print the links found and every rank's docker command, change nothing
# Extra arguments go to rank 0's `tensorfold-native serve` after the defaults, so they win.
# Setup: WORKER, WORKER2, WORKER3 (ssh targets, key-based) in scripts/local.sh: on a ring in its order.
# Settings, from the environment, scripts/local.sh or ./.env (defaults and their notes in scripts/config.sh):
#   serving  CONTEXT, POOL, PARALLEL, ROUND_ROWS, MAX_TOKENS, THINKING, REASONING_EFFORT, TEMPERATURE, TOP_P, TOP_K, DRAFTS, VISION, SERVED_NAME,
#            HOST, PORT, RDMA, GRAPHS, SIDE, PREFETCH, ENGRAM_AIO, ENGRAM_PACKED, ENGRAM_OWNED, ENGRAM_CACHE_MIB,
#            ENGRAM_INSTREAM, ROUND_MS, DRAFT_MS, SERVED_K
#   nodes    WORKER, WORKER2, WORKER3, FABRIC_PEER*, MASTER_ADDR, MASTER_PORT, SOCKET_IFNAME, ORDER, NCCL_CHANNELS,
#            NCCL_DEBUG
#   files    DERISKED, NVFP4, MODEL_ID, MODEL_REVISION, HF_CACHE, NFS_PATH, NFS_SERVER*, NFS_VOLUME, STATE_DIR, LOG_DIR, LOG_KEEP
#   engine   TF_REPO, TF_REF, KIT_REPO, KIT_REF, ZIG_VERSION, BASE_IMAGE, IMAGE, CONTAINER_NAME
#   setup    PREPARE (auto | 1 | 0), WAIT_TIMEOUT (seconds, default 2400), DRY_RUN=1
#   ranks    every TENSORFOLD_* and TF_DS_* variable you set goes to every rank
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
# the fabric this script serves: mesh (every pair of Sparks on one RoCE subnet: a switch); ./start-switchless.sh sets
# ring. scripts/nodes.sh stops when the links are the other kind and names the script that fits.
export FABRIC_EXPECT="${FABRIC_EXPECT:-mesh}"
source ./scripts/config.sh
source ./scripts/nodes.sh

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; }
WAIT_TIMEOUT="${WAIT_TIMEOUT:-2400}"
MODE=start
case "${1:-}" in
  restart) MODE=restart; shift ;;
  help|-h|--help) usage; exit 0 ;;
esac
DRY=0; [[ "${DRY_RUN:-0}" == 1 ]] && DRY=1
for v in CONTEXT POOL PARALLEL ROUND_ROWS MAX_TOKENS; do [[ "${!v}" =~ ^[1-9][0-9]*$ ]] || die "$v is a positive count, not ${!v}"; done
(( CONTEXT <= 1048576 )) || die "CONTEXT is at most 1048576 (the model's window), not $CONTEXT"
(( POOL >= CONTEXT )) || die "POOL ($POOL) holds every stream's positions together: at least CONTEXT ($CONTEXT)"
(( PARALLEL <= 16 )) || die "PARALLEL is 1 to 16, not $PARALLEL"
[[ "$PROMPT_CACHE_GIB" =~ ^(0|[0-9]+([.][0-9]+)?)$ ]] || die "PROMPT_CACHE_GIB is a non-negative number of GiB, not $PROMPT_CACHE_GIB"
[[ "$SERVED_K" =~ ^(solo|all|off)$ ]] || die "SERVED_K is solo, all or off, not $SERVED_K"
for v in THINKING DRAFTS VISION RDMA GRAPHS SIDE PREFETCH ENGRAM_AIO ENGRAM_PACKED ENGRAM_OWNED ENGRAM_INSTREAM RANK_LOCAL; do [[ "${!v}" =~ ^[01]$ ]] || die "$v is 0 or 1, not ${!v}"; done
[[ "$ENGRAM_CACHE_MIB" =~ ^[0-9]+$ ]] || die "ENGRAM_CACHE_MIB is a count of MiB, not $ENGRAM_CACHE_MIB"
[[ "$ENGRAM_INSTREAM" == 0 || ( "$ENGRAM_PACKED" == 1 && "$ENGRAM_AIO" == 1 ) ]] || die "ENGRAM_INSTREAM=1 needs ENGRAM_PACKED=1 and ENGRAM_AIO=1"
check_workers

# The operator's arguments are appended to rank 0's serve line, so they win over the environment: everything that
# compares against the launch must read the same final value the engine will — the last occurrence, in either form
# (`--flag value`, `--flag=value`). A flag given no value at all (`--context=`, a trailing `--context`) is the operator's
# mistake and is refused here, not by the engine once the ranks are launched.
EXTRA=("$@")
argv_flag() {  # sets ARGV_SEEN (the flag was given) and ARGV_VALUE (its last value: empty when it was given none)
  local name=$1 i=0 arg
  ARGV_SEEN=0; ARGV_VALUE=
  while (( i < ${#EXTRA[@]} )); do
    arg=${EXTRA[i]}
    if [[ "$arg" == "$name" ]]; then
      ARGV_SEEN=1; ARGV_VALUE=
      if (( i + 1 < ${#EXTRA[@]} )); then ARGV_VALUE=${EXTRA[i + 1]}; fi
    elif [[ "$arg" == "$name="* ]]; then
      ARGV_SEEN=1; ARGV_VALUE=${arg#*=}
    fi
    i=$((i + 1))
  done
  return 0
}
eff_flag() {  # the operator's last --flag, else the environment's value; `what` completes "needs a value: ..."
  local name=$1 env=$2 what=$3
  argv_flag "$name"
  if (( ARGV_SEEN )); then
    [[ -n "$ARGV_VALUE" ]] || die "$name needs a value: $what"
    printf '%s' "$ARGV_VALUE"
  else
    printf '%s' "$env"
  fi
}
EFF_CONTEXT=$(eff_flag --context "$CONTEXT" "a token count, or 0 for the model's window")
EFF_PARALLEL=$(eff_flag --parallel "$PARALLEL" "a stream count, or auto")
EFF_CACHE_GIB=$(eff_flag --prompt-cache-gib "$PROMPT_CACHE_GIB" "a non-negative number of GiB")
# The engine refuses these only after loading ~75 GiB on every Spark: refuse them before any container starts. A count is
# a plain decimal: digits only, no sign and no leading zero (Bash would read 0100 as octal 64), at most a few digits so no
# comparison here can wrap.
[[ "$EFF_CONTEXT" =~ ^(0|[1-9][0-9]{0,6})$ ]] || die "--context $EFF_CONTEXT: a plain decimal token count, or 0 for the model's window"
(( EFF_CONTEXT <= 1048576 )) || die "--context $EFF_CONTEXT is past the model's 1,048,576-token window"
# Only rank 0 takes these arguments; the followers are launched with CONTEXT and allocate buffers for it (ctx_cap in
# model.zig). A wider rank-0 window would ask them to hold positions they were never sized for, so a wider window has
# to be configured through CONTEXT for every rank.
if (( EFF_CONTEXT == 0 )); then
  (( CONTEXT >= 1048576 )) || die "--context 0 asks for the model's 1,048,576-token window, above CONTEXT ($CONTEXT), what the followers are launched with: set CONTEXT=1048576 for all four ranks instead"
else
  (( EFF_CONTEXT <= CONTEXT )) || die "--context $EFF_CONTEXT is above CONTEXT ($CONTEXT), what the followers are launched with and sized for: set CONTEXT for all four ranks instead"
fi
if [[ "$EFF_PARALLEL" == auto ]]; then
  # the engine's auto is 8 streams (server/cli.zig), and this backend reports stream_bytes = 0 (families/dsv41/native.zig),
  # so the CUDA budget never reduces them: below 8 slots the extra streams meet NoFreeSlot, which is not requeued.
  (( PARALLEL >= 8 )) || die "--parallel auto asks the engine for its 8 streams, above PARALLEL ($PARALLEL), what this launch and the shared pool were sized for: raise PARALLEL (in the environment or scripts/local.sh) to 8 or more, or pass a count of $PARALLEL or fewer"
else
  # the engine clamps 0 up to 1 and 4096 down at the top; 0 would also make the check expect 0 streams where the engine
  # reports 1. A count above PARALLEL is past what this launch and the shared pool were sized for: NoFreeSlot later.
  [[ "$EFF_PARALLEL" =~ ^[1-9][0-9]{0,3}$ ]] || die "--parallel $EFF_PARALLEL: a plain decimal stream count from 1, or auto (the engine reads 0 as 1)"
  (( EFF_PARALLEL <= PARALLEL )) || die "--parallel $EFF_PARALLEL is above PARALLEL ($PARALLEL), what this launch and the shared pool were sized for: raise PARALLEL (in the environment or scripts/local.sh) to run more streams"
fi
[[ "$EFF_CACHE_GIB" =~ ^(0|[0-9]+([.][0-9]+)?)$ ]] || die "--prompt-cache-gib $EFF_CACHE_GIB: a non-negative number of GiB"
cache_whole=${EFF_CACHE_GIB%%.*}
while [[ "$cache_whole" == 0?* ]]; do cache_whole=${cache_whole#0}; done  # 08 is 8 GiB, not Bash's octal error
# The engine refuses a budget at or past 2^34 GiB: refuse it before any container starts.
[[ ${#cache_whole} -le 11 ]] || die "--prompt-cache-gib $EFF_CACHE_GIB is past the engine's 2^34 GiB ceiling"
(( 10#$cache_whole < 17179869184 )) || die "--prompt-cache-gib $EFF_CACHE_GIB is past the engine's 2^34 GiB ceiling"
# The engine truncates the budget to whole bytes. A positive budget below one byte truncates to 0, which disables
# the store, while the check below reads the launch as one that asked for it: refuse the ambiguity, 0 is explicit.
if (( 10#$cache_whole == 0 )); then
  cache_bytes=$(awk -v g="$EFF_CACHE_GIB" 'BEGIN { print int(g * 1073741824) }')  # truncated, as the engine reads it
  (( cache_bytes > 0 )) || die "--prompt-cache-gib $EFF_CACHE_GIB is below one byte once the engine truncates it: pass 0 to disable the prompt-state store"
fi

B=$'\033[1m'; M=$'\033[1;35m'; G=$'\033[1;32m'; D=$'\033[2m'; R=$'\033[0m'
[[ -t 1 ]] || { B=; M=; G=; D=; R=; }
source ./scripts/banner.sh
echo
banner
printf '\n%s  Mia'"'"'s TensorFold Start Script%s\n' "$M" "$R"
printf '%s  %s · 4 x DGX Spark · %s at once · %s-token window · %s · port %s%s\n\n' "$D" "$MODEL_ID" "$PARALLEL" "$CONTEXT" "$MODEL_FORMAT" "$PORT" "$R"
STEPS=5
step() { printf '%s[%s/%s]%s %s%s%s\n' "$M" "$1" "$STEPS" "$R" "$B" "$2" "$R"; }

command -v docker >/dev/null || die "docker is not installed"
mkdir -p "$STATE_DIR" "$LOG_DIR"
exec 8>"$STATE_DIR/start.lock"
flock -n 8 || die "another ./start.sh is already running; wait for it to finish"
need_workers
(( DRY )) && log "DRY_RUN=1: printing the docker commands; nothing is stopped, started or prepared"

API_HOST="$HOST"; [[ "$HOST" == 0.0.0.0 || "$HOST" == "::" ]] && API_HOST=127.0.0.1
[[ "$API_HOST" != *:* ]] || API_HOST="[$API_HOST]"   # an IPv6 address in a URL needs brackets
URL="http://$API_HOST:$PORT"
running_here()   { [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)" == true ]]; }
running_worker() { [[ "$(worker "$1" docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)" == true ]]; }
all_running() { local i; running_here || return 1; for i in 1 2 3; do running_worker "$i" || return 1; done; }
if (( ! DRY )) && [[ "$MODE" == start ]] && all_running; then
  log "$CONTAINER_NAME is already running on all 4 Sparks (port $PORT): nothing to do. $(launcher "$FABRIC_EXPECT") restart restarts it, ./stop.sh stops it."
  exit 0
fi

# ---------------------------------------------------------------- 1. setup
step 1 "Setup: image, checkpoint, weight files and kit on spark1; NFS volumes on the workers"
need_prepare=0
case "${PREPARE:-auto}" in
  1) need_prepare=1 ;;
  0) ;;
  *) # (computed on its own: a failure inside $(...) would otherwise read as an empty line equal to an empty marker)
     state=$(prepared_state) || die "cannot tell whether the setup is ready: see the error above"
     [[ "$state" == "$(cat "$PREPARED_MARKER" 2>/dev/null)" ]] || need_prepare=1 ;;
esac
if (( need_prepare )); then
  if (( DRY )); then log "DRY_RUN: scripts/prepare.sh would run now (not ready yet)"
  else
    log "Not ready yet: running scripts/prepare.sh (the first time this builds the image, downloads $MODEL_SIZE and writes four ~80 GB weight files)"
    ./scripts/prepare.sh
  fi
else
  log "Ready: $IMAGE, $MODEL_ID @ $(snapshot_rev | cut -c1-8), its weight files and kit"
fi
docker image inspect "$IMAGE" >/dev/null 2>&1 || (( DRY )) || die "image $IMAGE missing: scripts/prepare.sh did not build it"
MODEL_ARG=$(model_in_container)

# ---------------------------------------------------------------- 2. checks
step 2 "Checks: links, previous server, port, memory"
detect_links
NODE_ORDER=$(node_order)
log "Fabric: $FABRIC; rendezvous $MASTER_ADDR:$MASTER_PORT; 2D nodes by NCCL rank: $NODE_ORDER"
for r in 0 1 2 3; do
  log "  rank $r ($( (( r == 0 )) && echo this node || worker_host "$r")): RoCE ${NODE_HCAS[r]} (GID ${NODE_GID[r]:-per device})$( (( r == 0 )) || echo ", NFS from ${LINK_HEAD_ADDR[r]:-?}")"
done
if (( ! DRY )); then
  if running_here || running_worker 1 || running_worker 2 || running_worker 3; then ./stop.sh; fi
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  for i in 1 2 3; do worker "$i" "docker rm -f '$CONTAINER_NAME' >/dev/null 2>&1; true"; done
  if ss -ltn "sport = :$PORT" 2>/dev/null | grep -q LISTEN; then die "port $PORT is already in use: $(ss -ltnp "sport = :$PORT" 2>/dev/null | tail -n +2)"; fi
  here_gb=$(free -g | awk '/^Mem:/ {print $7}')
  (( here_gb >= 100 )) || warn "only ${here_gb} GiB memory available here: the weights take ~75 GiB a Spark, the caches the rest"
  for i in 1 2 3; do
    g=$(worker "$i" "free -g | awk '/^Mem:/ {print \$7}'" 2>/dev/null || echo 0)
    (( g >= 100 )) || warn "only ${g} GiB memory available on $(worker_host "$i"): stop other GPU work there"
  done
fi

# ---------------------------------------------------------------- 3. launch
# What every rank's engine reads (HF_CACHE at /hf: this Spark's own, the workers' over NFS)
ENGINE_ENV=(TF_DS_2D_GU=parity "TF_DS_2D_ORDER=$NODE_ORDER" "TF_DS_ROUND_ROWS=$ROUND_ROWS" "TF_DS_STREAMS=$PARALLEL" "TF_DS_SERVED_K=$SERVED_K"
            "TF_DS_WINDOW=$POOL" "TF_DS_GRAPHS=$GRAPHS" "TF_DS_HC_SIDE=$SIDE" "TF_DS_L2_PREFETCH=$PREFETCH" "TF_DS_ENGRAM_AIO=$ENGRAM_AIO"
            "TF_DS_ROUND_MS=$ROUND_MS" "TF_DS_DRAFT_MS=$DRAFT_MS" "HF_HUB_OFFLINE=$HF_HUB_OFFLINE")
# Engram's packed shards (ENGRAM_PACKED=1, zig/patches/0003): every rank reads its own at /engram-packed
[[ "$ENGRAM_PACKED" != 1 ]] || ENGINE_ENV+=(TF_DS_ENGRAM_PACKED=/engram-packed "TF_DS_ENGRAM_OWNED=$ENGRAM_OWNED"
                                            "TF_DS_ENGRAM_CACHE_MIB=$ENGRAM_CACHE_MIB" "TF_DS_ENGRAM_INSTREAM=$ENGRAM_INSTREAM")
while IFS='=' read -r name _; do ENGINE_ENV+=("$name=${!name}"); done < <(env | grep -E '^(TENSORFOLD|TF_DS|TF_SSE)_[A-Z0-9_]+=' | grep -vE '^TF_DS_(2D_GU|2D_ORDER|ROUND_ROWS|STREAMS|SERVED_K|WINDOW|GRAPHS|HC_SIDE|L2_PREFETCH|ENGRAM_AIO|ENGRAM_PACKED|ENGRAM_OWNED|ENGRAM_CACHE_MIB|ENGRAM_INSTREAM|ROUND_MS|DRAFT_MS)=' || true)
envs() { local x; for x in "$@"; do printf -- '-e %q ' "$x"; done; }
RUN_ARGS="--gpus all --ipc=host --network host --ulimit memlock=-1 --ulimit stack=67108864 --cap-add IPC_LOCK --device /dev/infiniband"
DATA=/hf/$DATA_SUB
KIT=/hf/$KIT_SUB/kit
# the followers take the round's settings as flags only (tf-dsv41-lanes reads no TF_DS_GRAPHS & co.): without them they
# run every round ungraphed, with no side stream, prefetch or AIO, and rank 0 waits on them each round
FOLLOW="--pool $POOL --context $CONTEXT --engram $MODEL_ARG --token-map $DATA/dsv41_token_map.json --drafts $DRAFTS --round-rows $ROUND_ROWS --streams $PARALLEL"
FOLLOW+=" --graphs $GRAPHS --side $SIDE --prefetch $PREFETCH --aio $ENGRAM_AIO"
step 3 "Launch: container $CONTAINER_NAME, ranks 3 to 1 on the workers, then rank 0 here"
IFS=, read -r -a ORD <<<"$NODE_ORDER"
LOCAL_RANKS=()   # the workers that read their weight file from their own disk
for i in 3 2 1; do
  # its weight file from its own disk when the copy there (prepare.sh) matches this Spark's, else over NFS
  rank_dir="$DATA/rank-cache"; rank_mount=""
  rf=""; [[ "$RANK_LOCAL" != 1 ]] || rf=$(rank_file "${ORD[i]}")
  if [[ -n "$rf" ]] && worker "$i" "[[ \"\$(stat -c '%s %Y' $WORKER_RANK_DIR/${rf##*/} 2>/dev/null)\" == '$(rank_stamp "$rf")' ]]" 2>/dev/null; then
    rank_dir=/rank-local; rank_mount="-v $WORKER_RANK_DIR:/rank-local:ro"; LOCAL_RANKS=("$i" "${LOCAL_RANKS[@]}")
  elif [[ "$RANK_LOCAL" == 1 ]]; then
    warn "rank $i reads its weight file over NFS: no matching copy in $WORKER_RANK_DIR on $(worker_host "$i") (PREPARE=1 $(launcher "$FABRIC_EXPECT") restart makes one)"
  fi
  # Engram's tables from the worker's own disk (prepare.sh), the rest over NFS; with packed shards the tables' headers
  # come over NFS and the rows from the shards on the worker's disk
  if [[ "$ENGRAM_PACKED" == 1 ]]; then
    cmd="tf-dsv41-lanes $MODEL_ARG $rank_dir $i 4 ${LINK_HEAD_ADDR[i]:-$MASTER_ADDR} $MASTER_PORT $KIT $FOLLOW"
    engram_mount="-v $ENGRAM_PACKED_DIR:/engram-packed:ro"
  else
    cmd="tf-dsv41-lanes $MODEL_ARG $rank_dir $i 4 ${LINK_HEAD_ADDR[i]:-$MASTER_ADDR} $MASTER_PORT $KIT ${FOLLOW/--engram $MODEL_ARG/--engram /engram}"
    engram_mount="-v $WORKER_ENGRAM_DIR:/engram:ro"
  fi
  [[ "$RDMA" == 0 ]] || cmd+=" --rdma ${NODE_HCAS[i]}"   # this node's RoCE devices
  remote="mkdir -p \$HOME/.cache/dsv41-tensorfold && docker run -d --name $CONTAINER_NAME $RUN_ARGS -v $NFS_VOLUME:/hf:ro $engram_mount $rank_mount $(envs "${ENGINE_ENV[@]}") $(rank_nccl_env "$i") $IMAGE $cmd"
  if (( DRY )); then printf '[dry-run] rank %s on %s:\n  %s\n' "$i" "$(worker_host "$i")" "$remote"; continue; fi
  log "Rank $i on $(worker_host "$i"): $cmd"
  worker "$i" "$remote" >/dev/null || die "could not start rank $i on $(worker_host "$i")"
done
# rank 0 listens for each worker at the address that worker was given (its link's address of this node; a ring has
# no one address every cable reaches)
LISTEN="${LINK_HEAD_ADDR[1]:-$MASTER_ADDR},${LINK_HEAD_ADDR[2]:-$MASTER_ADDR},${LINK_HEAD_ADDR[3]:-$MASTER_ADDR}"
ENV0=(TF_TP_WORLD=4 "TF_TP_MASTER=$MASTER_ADDR" "TF_TP_LISTEN=$LISTEN" "TF_TP_PORT=$MASTER_PORT" "TF_DS_RANK_CACHE=$DATA/rank-cache" "TF_DS_KIT=$KIT"
      "TENSORFOLD_CUDA_KERNELS=$KIT/aot" "TF_DS_ENGRAM=$MODEL_ARG" "TF_DS_TOKEN_MAP=$DATA/dsv41_token_map.json")
SERVE=(tensorfold-native serve "$(served_in_container)" --backend cuda --host "$HOST" --port "$PORT" --name "$SERVED_NAME"
       --context "$CONTEXT" --parallel "$PARALLEL" --prompt-cache-gib "$PROMPT_CACHE_GIB" --max-tokens "$MAX_TOKENS" --temperature "$TEMPERATURE" --top-p "$TOP_P" --top-k "$TOP_K")
if [[ "$THINKING" == 1 ]]; then SERVE+=(--thinking); else SERVE+=(--no-thinking); fi
[[ -z "$REASONING_EFFORT" ]] || SERVE+=(--reasoning-effort "$REASONING_EFFORT")
# drafts off on rank 0 as on the followers (a follower without the drafter refuses a drafted round)
[[ "$DRAFTS" == 1 ]] || SERVE+=(--no-drafts)
SERVE+=("$@")
here=(docker run -d --name "$CONTAINER_NAME" $RUN_ARGS -v "$HF_CACHE:/hf:ro")
[[ "$ENGRAM_PACKED" != 1 ]] || here+=(-v "$(packed_dir_here):/engram-packed:ro")
[[ "$RDMA" == 0 ]] || ENV0+=("TF_RDMA_DEVICES=${NODE_HCAS[0]}")
for x in "${ENV0[@]}" "${ENGINE_ENV[@]}"; do here+=(-e "$x"); done
read -r -a nenv <<<"$(rank_nccl_env 0)"
here+=("${nenv[@]}" "$IMAGE" "${SERVE[@]}")
if (( DRY )); then printf '[dry-run] rank 0 here:\n  %s\n' "$(printf '%q ' "${here[@]}")"; exit 0; fi
log "Rank 0 here: ${SERVE[*]}"
"${here[@]}" >/dev/null

# ---------------------------------------------------------------- 4. load
# where the workers read their weight file (most of the ~75 GiB): over NFS, unless RANK_LOCAL=1 mounted their own copy
local_n=${#LOCAL_RANKS[@]}; local_list=$(IFS=,; echo "${LOCAL_RANKS[*]}")
if (( local_n == 3 )); then step 4 "Loading: ~75 GiB of weights on each Spark from its own disk, then the caches (a minute or two)"
elif (( local_n == 1 )); then step 4 "Loading: ~75 GiB of weights on each Spark over NFS (rank $local_list from its own disk), then the caches (a few minutes)"
elif (( local_n == 2 )); then step 4 "Loading: ~75 GiB of weights on each Spark over NFS (ranks $local_list from their own disks), then the caches (a few minutes)"
else step 4 "Loading: ~75 GiB of weights on each Spark over NFS, then the caches (a few minutes)"; fi
# Neither log follower may retain the parent launcher lock after it exits.
docker logs -f "$CONTAINER_NAME" 8>&- 2>&1 | sed -u "s/^/  ${D}│${R} /" 8>&- &
LOGS_PID=$!
trap 'kill $LOGS_PID 2>/dev/null || true' EXIT
fail() {
  kill $LOGS_PID 2>/dev/null || true
  for i in 1 2 3; do printf '%s── rank %s (%s): last log lines ──%s\n' "$D" "$i" "$(worker_host "$i")" "$R"; worker "$i" docker logs --tail 15 "$CONTAINER_NAME" 2>&1 | sed 's/^/  │ /'; done
  die "$1"
}
start=$SECONDS
until curl -sf --max-time 5 "$URL/v1/models" >/dev/null 2>&1; do
  running_here || fail "rank 0 exited (code $(docker inspect -f '{{.State.ExitCode}}' "$CONTAINER_NAME")) before it was ready"
  for i in 1 2 3; do running_worker "$i" || fail "rank $i on $(worker_host "$i") exited before the server was ready"; done
  (( SECONDS - start < WAIT_TIMEOUT )) || fail "not ready after ${WAIT_TIMEOUT}s (WAIT_TIMEOUT); the ranks are still running"
  sleep 5
done
kill $LOGS_PID 2>/dev/null || true
log "Server answered after $((SECONDS - start))s"

# ---------------------------------------------------------------- 4b. effective cache configuration
# Read-only: the engine's own account of the shared pool, the per-request limit, the streams and the snapshot budget.
# It makes no inference request, warms nothing and changes nothing. The expected values are the ones the engine was
# actually given (EFF_*: the environment with the operator's arguments applied), and a value the engine resolves
# itself (`--context 0`: the model's window; `--parallel auto`: the engine-selected streams) is reported, not
# compared. An old engine reports no cache group: that is "unavailable" (documented), not a mismatch. A consistent
# report that disagrees with this launch is a warning by default, never silent: the ranks are already serving and a
# supervisor that restarts on a non-zero exit would restart a healthy server. CACHE_STRICT=1 makes it fatal instead.
# A /health that is not HTTP 200, or that carries a top-level error, is read three times before it is called fatal
# (one refused read from a server that is otherwise answering is not a verdict): that test is the one step 5 requires
# of the smoke test (the status is the last line of the body, as in step 5).
for attempt in 1 2 3; do
  health_raw=$(curl -s --max-time 10 -w '\n%{http_code}' "$URL/health") || health_raw=""
  [[ "${health_raw##*$'\n'}" == 200 ]] && break
  (( attempt < 3 )) && sleep 2
  true
done
cache_rc=0
cache_meta=$(EXPECT_POOL="$POOL" EXPECT_CONTEXT="$EFF_CONTEXT" EXPECT_PARALLEL="$EFF_PARALLEL" EXPECT_CACHE_GIB="$EFF_CACHE_GIB" \
  python3 -c '
import json, os, sys
raw = sys.stdin.read().rstrip("\n")
body, _, code = raw.rpartition("\n")
if not code:
    body, code = raw, ""
if code != "200":
    print("health answered HTTP %s, not 200" % (code or "nothing")); sys.exit(3)
try:
    doc = json.loads(body)
except Exception:
    print("health is unreadable"); sys.exit(3)
if not isinstance(doc, dict):
    print("health is not a JSON object"); sys.exit(3)
if doc.get("error") is not None:  # the same test step 5 applies: {} or a blank string is an error, not absence
    print("health reported an error: %s" % str(doc["error"])[:160]); sys.exit(3)
c = (doc.get("live") or {}).get("cache")
if not isinstance(c, dict):
    print("this engine reports none"); sys.exit(4)
if c.get("consistent") is not True:
    print("the engine reports inconsistent accounting"); sys.exit(5)
pool = c.get("pool") or {}
st = c.get("prompt_state") or {}
cap, ctx, streams = pool.get("capacity_positions"), c.get("context_limit_positions"), c.get("stream_capacity")
budget, enabled = st.get("budget_bytes"), st.get("enabled")
want_pool = int(os.environ["EXPECT_POOL"])
raw_ctx = os.environ["EXPECT_CONTEXT"].strip()
raw_par = os.environ["EXPECT_PARALLEL"].strip()
want_ctx = None if raw_ctx in ("", "0") else int(raw_ctx)
want_streams = None if raw_par.lower() in ("", "auto") else int(raw_par)
want_gib = float(os.environ["EXPECT_CACHE_GIB"])
problems, chosen = [], []
if cap != want_pool: problems.append("pool %s != %s" % (cap, want_pool))
if want_ctx is None:
    chosen.append("context {:,} (the model'"'"'s window)".format(ctx or 0))
elif ctx != want_ctx:
    problems.append("context %s != %s" % (ctx, want_ctx))
if want_streams is None:
    chosen.append("%s streams (engine-selected)" % streams)
elif streams != want_streams:
    problems.append("streams %s != %s" % (streams, want_streams))
want_bytes = int(want_gib * (1 << 30))  # the engine truncates (@intFromFloat): 0.2 GiB is 214748364 bytes, not the ...365 a round gives
if want_gib > 0 and enabled is not True: problems.append("the prompt-state store is disabled")
elif want_gib > 0 and budget != want_bytes: problems.append("snapshot budget %s != %s" % (budget, want_bytes))
elif want_gib == 0 and enabled is True: problems.append("the prompt-state store is enabled, not 0 GiB")
summary = "shared pool {:,} positions · per request {:,} · {:d} streams · snapshots {:.0f} MiB (held {:.1f} MiB, {:d} entries)".format(
    cap or 0, ctx or 0, streams or 0, (budget or 0) / 1048576, (st.get("held_bytes") or 0) / 1048576, st.get("entries") or 0)
if chosen:
    summary += " · " + ", ".join(chosen)
print((summary if not problems else summary + "; MISMATCH: " + " ".join(problems)))
sys.exit(0 if not problems else 6)
' <<<"$health_raw" 2>/dev/null) || cache_rc=$?
case $cache_rc in
  0) log "Cache telemetry: $cache_meta" ;;
  4) log "Cache telemetry: unavailable ($cache_meta); showing no pool or snapshot figures" ;;
  3) die "health check failed after 3 reads: $cache_meta. The engine must answer HTTP 200 with no error object before its cache account is read." ;;
  *) if [[ "${CACHE_STRICT:-0}" == 1 ]]; then
       die "cache telemetry check failed: $cache_meta. CACHE_STRICT=1 makes this fatal; the ranks were launched and are still serving (./stop.sh stops them)."
     else
       warn "cache telemetry check failed: $cache_meta. This launch's effective configuration is unverified; the ranks serve (CACHE_STRICT=1 makes this fatal)."
     fi ;;
esac

# ---------------------------------------------------------------- 5. smoke test
step 5 "Smoke test: one chat completion through all 4 ranks"
# a pass needs HTTP 200, no error object and the assistant's reply text (curl -w adds the status as the last line)
if smoke=$(curl -s --max-time 300 -w '\n%{http_code}' "$URL/v1/chat/completions" -H 'Content-Type: application/json' \
             -d "{\"model\": \"$SERVED_NAME\", \"max_tokens\": 16, \"temperature\": 0, \"chat_template_kwargs\": {\"enable_thinking\": false}, \"messages\": [{\"role\": \"user\", \"content\": \"Reply with OK.\"}]}" |
           python3 -c 'import json,sys
def no(why): sys.exit(why)
body, _, code = sys.stdin.read().rpartition("\n")
code == "200" or no("HTTP " + code)
r = json.loads(body)
r.get("error") is None or no("an error reply")
m = r["choices"][0]["message"]
m.get("role", "assistant") == "assistant" or no("not the assistant")
c = (m.get("content") or "").strip()
c or no("no reply text")
len(c) < 8 or len(set(c)) > 2 or no("one repeated character")
print(repr(c[:40]) + ",", r["usage"]["completion_tokens"], "tokens")' 2>/dev/null); then
  log "OK: $smoke"
else
  fail "the smoke test request failed (an HTTP error, an error reply, no reply text, or one character repeated); the ranks are still running"
fi
# the address to print: HOST itself, or for a wildcard one of this Spark's own (0.0.0.0 binds IPv4 only: an IPv4 one)
if [[ "$HOST" == 0.0.0.0 ]]; then IP=$(hostname -I 2>/dev/null | tr ' ' '\n' | awk 'NF && !/:/ {print; exit}')
elif [[ "$HOST" == "::" ]]; then IP=$(hostname -I 2>/dev/null | awk '{print $1}')
else IP="$HOST"; fi
[[ "$IP" != *:* ]] || IP="[$IP]"
printf '\n%s  ✔ %s is now LIVE! on port %s%s\n\n' "$G" "$SERVED_NAME" "$PORT" "$R"
cat <<EOF
    API      http://${IP:-<spark-address>}:$PORT/v1   (model: $SERVED_NAME)
    Window   $CONTEXT tokens · $PARALLEL at once · $MODEL_FORMAT · drafts $( [[ "$DRAFTS" == 1 ]] && echo on || echo off) · images $( [[ "$VISION" == 1 ]] && echo on || echo off)
    Pool     $POOL shared positions · $PROMPT_CACHE_GIB GiB of prompt-state snapshots (separate from the pool)
    Logs     docker logs -f $CONTAINER_NAME$(for i in 1 2 3; do printf '   (rank %s: ssh %s docker logs -f %s)' "$i" "$(worker_host "$i")" "$CONTAINER_NAME"; done)
    Restart  $(launcher "$FABRIC_EXPECT") restart
    Stop     ./stop.sh

EOF

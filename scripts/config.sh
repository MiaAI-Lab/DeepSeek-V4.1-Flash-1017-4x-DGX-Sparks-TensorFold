# Shared settings for start.sh, stop.sh and scripts/*.sh. A setting's value comes from the first of these that sets it:
#   1. the environment: `PORT=9000 ./start.sh`
#   2. scripts/local.sh (this setup's own values, above all WORKER .. WORKER3; sourced as bash), then ./.env (KEY=value
#      lines, read, never run): both are yours, not the repository's; where both set a key, local.sh wins
#   3. the defaults below
_cfg_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -f "$_cfg_root/scripts/local.sh" ]]; then
  declare -A _cfg_env=()
  while IFS= read -r _n; do _cfg_env[$_n]=${!_n}; done < <(compgen -e)
  source "$_cfg_root/scripts/local.sh"
  # the environment wins over local.sh: put back any variable it had that local.sh changed
  for _n in "${!_cfg_env[@]}"; do [[ "${!_n-}" == "${_cfg_env[$_n]}" ]] || export "$_n=${_cfg_env[$_n]}"; done
  unset _cfg_env
fi
if [[ -f "$_cfg_root/.env" ]]; then
  while IFS= read -r _line || [[ -n "$_line" ]]; do
    [[ "$_line" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
    _key=${BASH_REMATCH[2]}; _value=${BASH_REMATCH[3]}
    if [[ "$_value" =~ ^\"([^\"]*)\"[[:space:]]*(#.*)?$ || "$_value" =~ ^\'([^\']*)\'[[:space:]]*(#.*)?$ ]]; then
      _value=${BASH_REMATCH[1]}
    else
      _value=${_value%%#*}; _value=${_value%"${_value##*[![:space:]]}"}
    fi
    [[ -n "${!_key+set}" ]] || export "$_key=$_value"
  done < "$_cfg_root/.env"
fi
unset _n _line _key _value

# ---------------------------------------------------------------- the four Sparks
# This machine serves NCCL rank 0 and the API; WORKER, WORKER2, WORKER3 (ssh targets, key-based) are NCCL ranks 1-3.
# Four Sparks without a switch are a ring: cable them spark1-spark2-spark3-spark4-spark1 and list the workers in that
# order. On a switch any order works.
TP=4
WORKER="${WORKER:-}"; FABRIC_PEER="${FABRIC_PEER:-}"; WORKER_HF_CACHE="${WORKER_HF_CACHE:-}"
WORKER2="${WORKER2:-}"; FABRIC_PEER2="${FABRIC_PEER2:-}"; WORKER_HF_CACHE2="${WORKER_HF_CACHE2:-}"
WORKER3="${WORKER3:-}"; FABRIC_PEER3="${FABRIC_PEER3:-}"; WORKER_HF_CACHE3="${WORKER_HF_CACHE3:-}"
MASTER_PORT="${MASTER_PORT:-29760}"  # the ranks' rendezvous port (keep it on the private links)
# The rendezvous address: this node's address every worker reaches (what its hostname resolves to, else its first one).
# On a ring the workers across it reach it through a neighbour (scripts/nodes.sh says which address each gets).
_ma=$(getent ahostsv4 "$(hostname)" 2>/dev/null | awk '$1 !~ /^127\./ {print $1; exit}' || true)
[[ -n "$_ma" ]] || _ma=$(hostname -I 2>/dev/null | awk '{print $1}' || true)
MASTER_ADDR="${MASTER_ADDR:-$_ma}"
SOCKET_IFNAME="${SOCKET_IFNAME:-}"     # NCCL bootstrap override; auto: TP2 link, TP>2 common fabric or default route
# The 2D split's node (role) each NCCL rank serves (the engine's TF_DS_2D_ORDER). auto: 0,1,3,2 on a ring (every row
# and column partner then on a cable of its own; NCCL's all-gathers go round the ring in rank order), 0,1,2,3 on a switch.
ORDER="${ORDER:-auto}"

# ---------------------------------------------------------------- the checkpoint, the engine, the kernel kit
# DeepSeek's own release, served as published: FP8 E4M3 linears (UE8M0 scales on 32 x 32 blocks), MXFP4 routed
# experts, bf16 elsewhere, the Engram tables in FP8 read from disk. ~510 GB, downloaded once on this Spark; the workers
# read it (and their weight files) from here over NFS.
# DERISKED=1 selects Blackfrost-AI's checkpoint; explicit MODEL_ID / MODEL_REVISION still win.
DERISKED="${DERISKED:-0}"
[[ "$DERISKED" =~ ^[01]$ ]] || { echo "DERISKED must be 0 or 1" >&2; exit 1; }
_default_model=deepseek-ai/DeepSeek-V4.1-Flash-1017
[[ "$DERISKED" != 1 ]] || _default_model=Blackfrost-AI/DeepSeek-V4.1-Flash-1017-DERISKED-FP8
MODEL_ID="${MODEL_ID:-$_default_model}"
_rev=""
case "$MODEL_ID" in
  deepseek-ai/DeepSeek-V4.1-Flash-1017) _rev=1bae0e1088993b9b82de08f59de5be5a34e42170 ;;
  Blackfrost-AI/DeepSeek-V4.1-Flash-1017-DERISKED-FP8) _rev=4bb7df57c6bf26caf40e8da905e4062aed088ea6 ;;
esac
unset _default_model
MODEL_REVISION="${MODEL_REVISION-$_rev}"
# ABLIT=1 grafts drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated's native-FP8 sidecar (attn.wo_b, layers 10-35,
# the overlay's "recipe A") onto the checkpoint before the rank caches are baked. Gated, free, automatic approval
# after accepting its terms at https://huggingface.co/drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated while
# logged in; export HF_TOKEN or `hf auth login` with that account. The vendor validates this exact sidecar against
# SGLang/vLLM, not TensorFold -- its own TensorFold-validated recipe needs a different, EXL3-quantized checkpoint
# this repo does not serve. Validate live (tools/validate.py, tools/needle_bench.py) before trusting a build of it.
ABLIT="${ABLIT:-0}"
[[ "$ABLIT" =~ ^[01]$ ]] || { echo "ABLIT must be 0 or 1" >&2; exit 1; }
ABLIT_ID=drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated
ABLIT_REVISION="${ABLIT_REVISION:-de412d2249632903dace23399526e00a145aada5}"
ABLIT_SIDECAR=wo_b_l10_35_1017.safetensors
# The engine: BertholomusAI's TensorFold fork, branch deepseek-v41-zig-tp4 (the DeepSeek-V4.1-Flash family on
# TensorFold's native engine), at the commit of their four-node recipe v0.1.1, plus zig/patches (zig/PORT_REF).
TF_REPO="${TF_REPO:-https://github.com/bertholomus/TensorFold.git}"
TF_REF="${TF_REF:-59402086381a42c74d53053443224eec2790b4c5}"
# The family's kernel kit (its Triton kernels as captured binaries, Engram's constants, the vision tower's attention
# kernel): the kit/ of BertholomusAI's four-node recipe at v0.1.1, checked against its MANIFEST; the RoPE tables and
# the image routing bias are made from the checkpoint by its verify_kit.sh.
KIT_REPO="${KIT_REPO:-https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp4-4xgb10.git}"
KIT_REF="${KIT_REF:-6033ea48a27004b2715db82190e2a61f33411d92}"
ZIG_VERSION="${ZIG_VERSION:-0.17.0}"
ZIG_SHA256="${ZIG_SHA256:-9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8}"   # zig-aarch64-linux-0.17.0.tar.xz
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
image_hash() {
  ( cd "$_cfg_root" || exit 1
    cat zig/patches/*.patch 2>/dev/null
    cat zig/Dockerfile
    echo "$TF_REPO $TF_REF zig-$ZIG_VERSION $ZIG_SHA256 $BASE_IMAGE"
  ) | sha256sum | cut -c1-12
}
IMAGE="${IMAGE:-tensorfold-dsv41:native-$(image_hash)}"
CONTAINER_NAME="${CONTAINER_NAME:-dsv41-1017-tf}"   # the same name on every Spark

# ---------------------------------------------------------------- serving
SERVED_NAME="${SERVED_NAME:-DeepSeek-V4.1-Flash-1017}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8888}"
# A request's positions at most (prompt and reply), and the positions all requests share (the streams' extents in one
# pool). The model's window is 1,048,576 tokens.
CONTEXT="${CONTEXT:-1048576}"
POOL="${POOL:-2097152}"
PARALLEL="${PARALLEL:-16}"             # requests at once (the engine's streams)
ROUND_ROWS="${ROUND_ROWS:-48}"         # a decode round's rows at most (drafts of every stream together)
MAX_TOKENS="${MAX_TOKENS:-32768}"      # a reply's budget when a request sets none
THINKING="${THINKING:-1}"              # think first unless a request turns it off
# The reasoning effort when a request names none (the chat template's levels: low 50, high 75, max 100 on its 1-100
# scale; a request's own reasoning_effort wins). Without it the template's own default is high.
REASONING_EFFORT="${REASONING_EFFORT:-max}"
# The model card's recommended sampling (temperature 1.0, top_p 1.0) as the server's defaults, from the 20 likeliest
# tokens: the family's exact sampler needs a top-k (top_k 0, the nucleus alone, is not in it; BertholomusAI serve 20).
# A request's own values win.
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:-20}"
DRAFTS="${DRAFTS:-1}"                  # the checkpoint's own DSpark draft blocks (exact: drafted replies equal plain)
VISION="${VISION:-1}"                  # images in user turns and tool results (rank 0 runs the vision tower)
# The engine's settings as BertholomusAI's served launch has them (their README "The served settings"): CUDA graphs,
# the mixes' side stream, the paced L2 prefetch, Engram reads by Linux AIO, the lane core's measured round costs.
# The decode-size exchanges over RDMA (RoCE) instead of NCCL, on each node's RoCE devices (scripts/nodes.sh finds
# them); on a ring the Spark across is reached through its two neighbours, which relay half each (zig/patches: the
# ring-aware RDMA rings; the engine's own assumed a switch).
RDMA="${RDMA:-1}"
GRAPHS="${GRAPHS:-1}"
SIDE="${SIDE:-1}"
PREFETCH="${PREFETCH:-1}"
ENGRAM_AIO="${ENGRAM_AIO:-1}"
# Engram's rows from packed shards (zig/patches/0003, off by default): each Spark keeps the tables' rows as records (a
# row's FP8 values and its scales side by side, so a row is one read instead of two), written once by
# tf-dsv41-engram-pack, with a row cache in memory and the FP8 decode on the GPU; the rows a layer reads are the same
# bytes. ENGRAM_OWNED=1: each Spark keeps and reads only its quarter of the rows (~51 GB a Spark), and one exchange a
# layer fills in the rest; 0: every Spark keeps the whole tables (~203 GB). ENGRAM_CACHE_MIB: the row cache a Spark (0:
# none). ENGRAM_INSTREAM=1: a round's wait for its rows runs as a host step on the GPU stream instead of holding the host
# (needs ENGRAM_AIO=1).
ENGRAM_PACKED="${ENGRAM_PACKED:-0}"
ENGRAM_OWNED="${ENGRAM_OWNED:-1}"
ENGRAM_CACHE_MIB="${ENGRAM_CACHE_MIB:-2048}"
ENGRAM_INSTREAM="${ENGRAM_INSTREAM:-0}"
# A decode round's ms by its rows (1 .. ROUND_ROWS), the lane core's price of a row when it chooses how many drafts
# each stream verifies (TF_DS_ROUND_MS): measured on this build (2026-10-24, A1): a round
# of N streams x 1 row costs 21.5, 36.0, 53.4, 83.1 ms at N = 1, 4, 8, 16 (host steps included), linear between, past
# 16 the bandwidth model's falling slope (+3.3 ms a row to 32, +2.7 to 48). The EXL3-era table this replaces (14.5 ...
# 81.7) priced rows at half their cost, so the lane over-drafted at 4-16 streams.
ROUND_MS="${ROUND_MS:-21.5,26.3,31.1,36.0,40.3,44.7,49.0,53.4,57.1,60.8,64.5,68.2,72.0,75.7,79.4,83.1,86.4,89.7,93.0,96.3,99.6,102.9,106.2,109.5,112.8,116.1,119.4,122.7,126.0,129.3,132.6,135.9,138.6,141.3,144.0,146.7,149.4,152.1,154.8,157.5,160.2,162.9,165.6,168.3,171.0,173.7,176.4,179.1}"
# A round's ms past its forward that the one-stream chooser adds to every choice of drafts (TF_DS_DRAFT_MS): with the
# table above, 25 keeps one stream's choice where the old table had it (the measured A/B: 12 and 18 lost 0.5-3 % at c1).
DRAFT_MS="${DRAFT_MS:-25}"
# The drafts a stream verifies at most (the engine's TF_DS_SERVED_K): solo (the served lane's own choice of k, from
# ROUND_MS and its draft ms, when one stream is live; more streams keep the lane core's allocation), all (that choice at
# any number of streams), off (the lane core's allocation alone). Replies are the same either way; only speed changes.
SERVED_K="${SERVED_K:-solo}"

# ---------------------------------------------------------------- files
HF_CACHE="${HF_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}}"
# What the ranks read besides the checkpoint (each worker over NFS): the four weight files, Engram's token map and the
# kernel kit, under HF_CACHE so that one export serves them all.
DATA_SUB="tensorfold-dsv41/${MODEL_REVISION:0:8}"
[[ "$ABLIT" != 1 ]] || DATA_SUB+="-ablit-${ABLIT_REVISION:0:8}"
DATA_DIR="$HF_CACHE/$DATA_SUB"
KIT_SUB="tensorfold-dsv41/kit-${KIT_REF:0:8}"
# The workers read this Spark's HF_CACHE over NFS (the recipe's only mode: one copy of ~510 GB, on spark1): a read-only
# docker volume NFS_VOLUME on each worker that prepare.sh creates. This Spark must export NFS_PATH (the path of HF_CACHE
# in its export, default /huggingface) to every worker's address on its link. NFS_SERVER, NFS_SERVER2, NFS_SERVER3: the
# head's address each worker mounts from (default: the head's address on its link, found by scripts/nodes.sh).
NFS_PATH="${NFS_PATH:-/huggingface}"
NFS_SERVER="${NFS_SERVER:-}"; NFS_SERVER2="${NFS_SERVER2:-}"; NFS_SERVER3="${NFS_SERVER3:-}"
NFS_VOLUME="${NFS_VOLUME:-dsv41-hf}"
# Engram's tables (the checkpoint's shards 47 and 48, ~203 GB) on each worker's own disk: a decode round reads ~70
# random 4 KiB rows from them, which over NFS took ~10 ms each and most of the round. prepare.sh copies them there once
# (from the NFS volume); the main weights and the weight files still load over NFS. A path on the worker.
WORKER_ENGRAM_DIR="${WORKER_ENGRAM_DIR:-\$HOME/.cache/dsv41-tensorfold/engram-${MODEL_REVISION:0:8}}"
# With ENGRAM_PACKED=1 the packed shards replace that copy, on every Spark's own disk (this one's too): a path on each.
ENGRAM_PACKED_DIR="${ENGRAM_PACKED_DIR:-\$HOME/.cache/dsv41-tensorfold/engram-packed-${MODEL_REVISION:0:8}}"
packed_dir_here() { echo "${ENGRAM_PACKED_DIR//\$HOME/$HOME}"; }
# The nodes the packed shards are cut for: the 2D split's four (ENGRAM_OWNED=1), or one (the whole tables).
packed_nodes() { if [[ "$ENGRAM_OWNED" == 1 ]]; then echo 4; else echo 1; fi; }
engram_shards() { (cd "$(model_cache_dir)/snapshots/$(snapshot_rev)" && python3 -c "
import json; m = json.load(open('model.safetensors.index.json'))['weight_map']
print(' '.join(sorted({f for k, f in m.items() if '.engram.' in k})))"); }
STATE_DIR="${STATE_DIR:-$HOME/.local/state/dsv41-tensorfold}"   # locks, the prepared marker, NCCL graph files
LOG_DIR="${LOG_DIR:-$HOME/.cache/dsv41-tensorfold/logs}"
LOG_KEEP="${LOG_KEEP:-10}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# Colours only on a terminal.
_c() { [[ -t "$1" ]] && printf '\033[%sm' "$2" || true; }
log()  { printf '%s[%s]%s %s\n' "$(_c 1 '1;36')" "$(basename "$0")" "$(_c 1 0)" "$*"; }
warn() { printf '%s[%s] WARN:%s %s\n' "$(_c 2 '1;33')" "$(basename "$0")" "$(_c 2 0)" "$*" >&2; }
die()  { printf '%s[%s] ERROR:%s %s\n' "$(_c 2 '1;31')" "$(basename "$0")" "$(_c 2 0)" "$*" >&2; exit 1; }

[[ -z "${HF_TOKEN:-}" ]] || export HF_TOKEN
model_cache_dir() { echo "$HF_CACHE/hub/models--${MODEL_ID//\//--}"; }
# The snapshot this setup serves: the pin, else what refs/main names here.
snapshot_rev() { local rev=$MODEL_REVISION; [[ -n "$rev" ]] || rev=$(cat "$(model_cache_dir)/refs/main" 2>/dev/null); echo "$rev"; }
# The checkpoint as the containers see it (HF_CACHE mounted at /hf on every Spark). ABLIT=1: the overlaid view
# prepare.sh builds at $DATA_SUB/checkpoint (every stock file linked, plus the Keys sidecar).
model_in_container() {
  if [[ "$ABLIT" == 1 ]]; then echo "/hf/$DATA_SUB/checkpoint"
  else echo "/hf/hub/models--${MODEL_ID//\//--}/snapshots/$(snapshot_rev)"; fi
}
# The folder rank 0 serves (prepare.sh): the checkpoint's files linked, with the recipe's chat template.
served_in_container() { echo "/hf/$DATA_SUB/model"; }

# What scripts/prepare.sh last left ready (start.sh runs it again when this line changes): with where the checkpoint is
# (HF_CACHE), how the workers mount it (NFS_PATH, NFS_VOLUME, NFS_SERVER*) and the chat template it copies into the
# served folder, each of which needs prepare.sh's steps again when it changes. Each field is shell-quoted (%q, in the C
# locale so the line does not depend on it), so no two settings give the same line.
PREPARED_MARKER="$STATE_DIR/prepared"
prepared_state() {
  local i tpl f
  tpl=$(sha256sum "$_cfg_root/model/chat_template.jinja" | cut -c1-64) && [[ -n "$tpl" ]] ||
    die "cannot hash model/chat_template.jinja (sha256sum)"
  f=("model=$MODEL_ID@$(snapshot_rev)" "image=$(image_ident "$IMAGE")" "kit=$KIT_REF" "order=$ORDER" "engram=$WORKER_ENGRAM_DIR" "ablit=$ABLIT")
  [[ "$ENGRAM_PACKED" != 1 ]] || f+=("packed=$ENGRAM_PACKED_DIR/$(packed_nodes)")
  f+=("hf=$HF_CACHE" "nfs_path=$NFS_PATH" "nfs_volume=$NFS_VOLUME" "template=$tpl")
  for i in 1 2 3; do f+=("w$i=$(worker_host "$i")" "nfs$i=$(wval NFS_SERVER "$i")"); done
  (export LC_ALL=C; printf '%q ' "${f[@]}"; echo)   # (a subshell: the caller's locale is left as it was)
}
# prepare.sh's last step: computed first and swapped in whole, so a failure leaves the previous marker as it was
write_prepared_marker() {
  local state
  state=$(prepared_state) || return 1
  printf '%s\n' "$state" > "$PREPARED_MARKER.new" && mv -f "$PREPARED_MARKER.new" "$PREPARED_MARKER"
}

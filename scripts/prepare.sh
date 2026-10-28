#!/usr/bin/env bash
# Prepare the four Sparks to serve deepseek-ai/DeepSeek-V4.1-Flash-1017 with TensorFold's native engine. Everything
# lands on this Spark (spark1); the workers read it over NFS:
#   1. preflight: docker and the GPU on every Spark, key-based ssh, the links (a switch, or a ring in rank order)
#   2. the engine image on this Spark (zig/Dockerfile: BertholomusAI's TensorFold fork at TF_REF plus zig/patches, on
#      BASE_IMAGE), then on each worker (docker save | docker load: only what it lacks is stored)
#   3. the checkpoint into HF_CACHE (~510 GB, resumable), at its pinned revision
#   4. the kernel kit: BertholomusAI's kit/ at KIT_REF, its RoPE tables and image bias made from the checkpoint, every
#      file checked against its MANIFEST (the image bias is 1017's own: the one file that differs from theirs)
#   5. Engram's token map and the four nodes' weight files (one each, ~80 GB) from the checkpoint, no GPU, and the
#      folder rank 0 serves: the checkpoint's files linked, beside model/chat_template.jinja (the release ships its
#      chat format as encoding/encoding.py, which the server cannot run)
#   6. a read-only NFS volume of this Spark's HF_CACHE on each worker, checked: every file a rank reads, by size
#   7. Engram's tables on each worker's disk, or with ENGRAM_PACKED=1 each Spark's packed shards (tf-dsv41-engram-pack)
# ./start.sh runs this by itself when needed. Safe to re-run: every step skips work that is already done.
# Pass --rebuild to rebuild the image from scratch.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
source ./scripts/config.sh
source ./scripts/nodes.sh

REBUILD=0
for arg in "$@"; do
  case "$arg" in
    --rebuild) REBUILD=1 ;;
    -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
    *) die "unknown argument: $arg" ;;
  esac
done

# ---------------------------------------------------------------- 1. preflight
mkdir -p "$STATE_DIR" "$HF_CACHE/hub"
exec 9>"$STATE_DIR/prepare.lock"
flock -n 9 || die "another prepare.sh is already running; wait for it or stop it: pgrep -af prepare.sh"
log "Preflight checks on all 4 Sparks"
command -v docker >/dev/null || die "docker is not installed"
docker info >/dev/null 2>&1 || die "cannot talk to the docker daemon (is your user in the docker group?)"
nvidia-smi -L >/dev/null 2>&1 || warn "nvidia-smi failed on this node: is the NVIDIA driver working?"
check_workers
for i in 1 2 3; do
  need_worker "$i"
  worker "$i" 'docker info >/dev/null 2>&1' || die "$(worker_host "$i") cannot talk to its docker daemon (docker group?)"
  worker "$i" 'nvidia-smi -L >/dev/null 2>&1' || warn "nvidia-smi failed on $(worker_host "$i")"
done
detect_links
log "Fabric: $FABRIC (2D nodes by NCCL rank: $(node_order))"
for i in 1 2 3; do log "  worker $i ($(worker_host "$i")): link ${LINK_HEAD_ADDR[i]:-?} <-> ${LINK_WORKER_ADDR[i]:-?}; NFS from $(nfs_server "$i")"; done

# ---------------------------------------------------------------- 2. the image
hash=$(image_hash)
built=$(docker image inspect -f '{{index .Config.Labels "tf.patches"}}' "$IMAGE" 2>/dev/null || true)
if [[ $REBUILD -eq 1 || "$built" != "$hash" ]]; then
  docker image inspect "$BASE_IMAGE" >/dev/null 2>&1 || { log "Pulling $BASE_IMAGE"; docker pull "$BASE_IMAGE" >/dev/null; }
  log "Building $IMAGE (TensorFold ${TF_REF:0:12} with $(compgen -G 'zig/patches/*.patch' | wc -l) patches, Zig $ZIG_VERSION; ~5 min)"
  ctx=$(mktemp -d "$STATE_DIR/build.XXXXXX")
  trap 'rm -rf -- "$ctx"' EXIT
  mkdir -p "$ctx/zig"; cp -r zig/patches "$ctx/zig/patches"; cp zig/Dockerfile "$ctx/Dockerfile"
  nocache=(); [[ $REBUILD -eq 1 ]] && nocache=(--no-cache)
  docker build "${nocache[@]}" -t "$IMAGE" -f "$ctx/Dockerfile" --build-arg BASE_IMAGE="$BASE_IMAGE" \
    --build-arg TF_REPO="$TF_REPO" --build-arg TF_REF="$TF_REF" --build-arg ZIG_VERSION="$ZIG_VERSION" \
    --build-arg ZIG_SHA256="$ZIG_SHA256" --build-arg PATCHES_HASH="$hash" "$ctx"
  rm -rf -- "$ctx"; trap - EXIT
else
  log "Image $IMAGE already built (hash $hash; --rebuild forces it)"
fi
ident=$(image_ident "$IMAGE")
for i in 1 2 3; do
  [[ "$(worker_image_ident "$i" "$IMAGE")" != "$ident" ]] || continue
  log "Copying $IMAGE to $(worker_host "$i") (docker save | docker load)"
  docker save "$IMAGE" | worker "$i" docker load >/dev/null
  [[ "$(worker_image_ident "$i" "$IMAGE")" == "$ident" ]] || die "$(worker_host "$i")'s $IMAGE differs from this Spark's"
done
log "Image $IMAGE identical on all 4 Sparks"

# ---------------------------------------------------------------- 3. the checkpoint
command -v hf >/dev/null || die "the Hugging Face CLI ('hf') is not installed on this Spark: pip install -U huggingface_hub"
log "Downloading $MODEL_ID${MODEL_REVISION:+ @ ${MODEL_REVISION:0:8}} into $HF_CACHE/hub (~510 GB the first time; resumes)"
if ! hf download "$MODEL_ID" ${MODEL_REVISION:+--revision "$MODEL_REVISION"} --cache-dir "$HF_CACHE/hub" >/dev/null; then
  [[ -n "$MODEL_REVISION" && -f "$(model_cache_dir)/snapshots/$MODEL_REVISION/model.safetensors.index.json" ]] ||
    die "$MODEL_ID: the download failed"
  warn "$MODEL_ID: could not reach Hugging Face; using the snapshot already here"
fi
SNAP="$(model_cache_dir)/snapshots/$(snapshot_rev)"
[[ -f "$SNAP/model.safetensors.index.json" ]] || die "$MODEL_ID: no complete snapshot at $SNAP"
log "Checkpoint: $SNAP ($(du -shL "$SNAP" | cut -f1))"
# a container's view of HF_CACHE (the snapshot's files are links into ../../blobs)
in_hf() { docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$HF_CACHE:/hf" "$@"; }

# ---------------------------------------------------------------- 4. the kit
KIT_DIR="$HF_CACHE/$KIT_SUB"
if [[ ! -d "$KIT_DIR/.git" ]]; then
  log "Fetching the kernel kit: $KIT_REPO @ ${KIT_REF:0:8}"
  rm -rf "$KIT_DIR"; git init -q "$KIT_DIR"
  git -C "$KIT_DIR" remote add origin "$KIT_REPO"
  git -C "$KIT_DIR" fetch -q --depth 1 origin "$KIT_REF"
  git -C "$KIT_DIR" checkout -q --detach FETCH_HEAD
fi
[[ "$(git -C "$KIT_DIR" rev-parse HEAD)" == "$KIT_REF" ]] || die "$KIT_DIR is not at KIT_REF ${KIT_REF:0:8}: remove it and run this again"
log "Checking the kit (its RoPE tables and image bias made from the checkpoint; CPU only)"
kit_out=$(IMAGE="$BASE_IMAGE" "$KIT_DIR/kit/verify_kit.sh" "$SNAP" "$SNAP" 2>&1 | tail -1) || true
# 1017's image routing bias is its own, not the original release's: the one file allowed to differ from MANIFEST
[[ "$kit_out" == *"356 of 356 files match"* || "$kit_out" == *"355 of 356 files match"*"1 differ, 0 missing"* ]] ||
  die "the kit does not check out: $kit_out"
grep -q "DIFFERS (bias): vision/gate_bias_vl.safetensors" <(IMAGE="$BASE_IMAGE" "$KIT_DIR/kit/verify_kit.sh" "$SNAP" 2>&1) ||
  [[ "$kit_out" == *"356 of 356"* ]] || die "a kit file other than the image bias differs from MANIFEST"
log "Kit: $kit_out (the image bias is 1017's own)"

# ---------------------------------------------------------------- 4.5 the Keys abliteration overlay (ABLIT=1)
if [[ "$ABLIT" == 1 ]]; then
  warn "ABLIT=1: $ABLIT_ID is gated. You must have accepted its terms at https://huggingface.co/$ABLIT_ID while" \
       "logged in, with a Hugging Face read token for that account: export HF_TOKEN=hf_... or hf auth login."
  log "Downloading the Keys overlay's native-FP8 sidecar ($ABLIT_SIDECAR, ~1.1 GB) from $ABLIT_ID @ ${ABLIT_REVISION:0:8}"
  hf download "$ABLIT_ID" --revision "$ABLIT_REVISION" --include "$ABLIT_SIDECAR" --cache-dir "$HF_CACHE/hub" >/dev/null ||
    die "$ABLIT_ID: the download failed (is its gate accepted by this HF_TOKEN's account?)"
  ABLIT_DIR="$DATA_DIR/checkpoint"
  if [[ ! -e "$ABLIT_DIR/$ABLIT_SIDECAR" ]]; then
    log "Building the overlaid checkpoint view ($ABLIT_DIR): every stock file linked, plus $ABLIT_SIDECAR" \
        "(our engine never reads model.safetensors.index.json: checkpoint.zig's Shards.open scans every" \
        "*.safetensors file in a directory by its own header and keeps the last one seen, in sorted-path order," \
        "for a given tensor name -- \"wo_b_...\" sorts after every \"model-...\" shard, so the overlay's 26 layers" \
        "of attn.wo_b win)"
    rm -rf "$ABLIT_DIR.new" && mkdir -p "$ABLIT_DIR.new"
    for f in "$SNAP"/*; do ln -s "../../../hub/models--${MODEL_ID//\//--}/snapshots/$(snapshot_rev)/$(basename "$f")" "$ABLIT_DIR.new/"; done
    ln -s "../../../hub/models--${ABLIT_ID//\//--}/snapshots/$ABLIT_REVISION/$ABLIT_SIDECAR" "$ABLIT_DIR.new/$ABLIT_SIDECAR"
    rm -rf "$ABLIT_DIR" && mv "$ABLIT_DIR.new" "$ABLIT_DIR"
  fi
  log "Checkpoint for baking: $ABLIT_DIR (stock + $ABLIT_SIDECAR grafted onto attn.wo_b, layers 10-35)"
fi

# ---------------------------------------------------------------- 5. token map and weight files
mkdir -p "$DATA_DIR/rank-cache"
SNAP_C=$(model_in_container)
if [[ ! -s "$DATA_DIR/dsv41_token_map.json" ]]; then
  log "Engram's token map from the checkpoint's tokenizer"
  in_hf --entrypoint python3 "$IMAGE" /opt/tensorfold/tools/inputs/make_token_map.py "$SNAP_C" "/hf/$DATA_SUB/dsv41_token_map.json" >/dev/null
fi
for g in 0 1 2 3; do
  # the file the engine looks for: its name carries the index's hash (tf-dsv41-rank-cache --index-only)
  want=$(in_hf --entrypoint bash "$IMAGE" -c "tf-dsv41-rank-cache $SNAP_C /tmp/x $g 4 --parity --index-only | sha256sum | cut -c1-20")
  if [[ -f "$DATA_DIR/rank-cache/rank${g}of4-$want.bin" ]]; then log "Weight file of node $g: ready"; continue; fi
  rm -f "$DATA_DIR/rank-cache/rank${g}of4-"*.bin "$DATA_DIR/rank-cache/.rank${g}of4.partial"
  log "Weight file of node $g from the checkpoint (~80 GB, a few minutes)"
  in_hf --entrypoint tf-dsv41-rank-cache "$IMAGE" "$SNAP_C" "/hf/$DATA_SUB/rank-cache" "$g" 4 --parity >/dev/null
done

# the served folder: every file of the snapshot (or, with ABLIT=1, the overlaid view above) as a relative link (it
# resolves under /hf in a container too), and the recipe's chat template
VIEW="$DATA_DIR/model"
rm -rf "$VIEW.new" && mkdir -p "$VIEW.new"
if [[ "$ABLIT" == 1 ]]; then
  for f in "$ABLIT_DIR"/*; do ln -s "../checkpoint/$(basename "$f")" "$VIEW.new/"; done
else
  for f in "$SNAP"/*; do ln -s "../../../hub/models--${MODEL_ID//\//--}/snapshots/$(snapshot_rev)/$(basename "$f")" "$VIEW.new/"; done
fi
cp model/chat_template.jinja "$VIEW.new/chat_template.jinja"
rm -rf "$VIEW" && mv "$VIEW.new" "$VIEW"
[[ -s "$VIEW/config.json" ]] || die "the served folder $VIEW does not reach the checkpoint"
log "Served folder: $VIEW (the checkpoint linked, the recipe's chat template)"

# ---------------------------------------------------------------- 6. the workers' NFS volumes
manifest() {  # the files a rank reads, with their sizes, as paths under HF_CACHE
  (cd "$HF_CACHE" && find -L "hub/models--${MODEL_ID//\//--}/snapshots/$(snapshot_rev)" "$DATA_SUB" "$KIT_SUB/kit" \
     -path '*/.git' -prune -o -type f -printf '%p %s\n' | LC_ALL=C sort)
}
want=$(manifest)
for i in 1 2 3; do
  ensure_nfs_volume "$i"
  if ! out=$(worker_nfs "$i" true 2>&1); then
    die "$(worker_host "$i") cannot mount this Spark's :$NFS_PATH from $(nfs_server "$i") over NFS ($NFS_VOLUME): $(tail -1 <<<"$out")
    This Spark must export HF_CACHE (as $NFS_PATH) read-only to that worker's address on its link (README: NFS)"
  fi
  have=$(worker "$i" "docker run --rm --network none -v '$NFS_VOLUME:/hf:ro' -w /hf --entrypoint find '$BASE_IMAGE' -L 'hub/models--${MODEL_ID//\//--}/snapshots/$(snapshot_rev)' '$DATA_SUB' '$KIT_SUB/kit' -path '*/.git' -prune -o -type f -printf '%p %s\n'" 2>/dev/null | LC_ALL=C sort || true)
  [[ "$have" == "$want" ]] || { diff <(printf '%s\n' "$want") <(printf '%s\n' "$have") | head -10 >&2
    die "$(worker_host "$i") does not see every file over NFS ($NFS_VOLUME)"; }
  log "Worker $i ($(worker_host "$i")) reads the checkpoint, its weight file and the kit over NFS"
done

# ---------------------------------------------------------------- 7. Engram's tables on each worker's disk
SNAP_REL="hub/models--${MODEL_ID//\//--}/snapshots/$(snapshot_rev)"
read -r -a shards <<<"$(engram_shards)"
(( ${#shards[@]} )) || die "the checkpoint's index names no Engram tables"
need=0; for f in "${shards[@]}"; do need=$(( need + $(stat -L -c %s "$SNAP/$f") )); done
if [[ "$ENGRAM_PACKED" == 1 ]]; then
  # ENGRAM_PACKED=1: packed shards on every Spark's own disk instead (zig/patches/0003), cut for the node rank r serves
  # (TF_DS_2D_ORDER) or whole; tf-dsv41-engram-pack keeps a shard that checks out, so a re-run writes only what is missing
  IFS=, read -r -a ord <<<"$(node_order)"
  nodes=$(packed_nodes)
  pneed=$(( need / nodes + 8192 ))
  for r in 0 1 2 3; do
    node=0; [[ "$nodes" == 1 ]] || node=${ord[r]}
    pack="docker run --rm --network none --user \$(id -u):\$(id -g) -e HOME=/tmp"
    if (( r == 0 )); then
      dir=$(packed_dir_here)
      mkdir -p "$dir"
      have=0; for f in "$dir"/engram-l*-n"${node}of${nodes}".tfe; do [[ -f "$f" ]] && have=$(( have + $(stat -c %s "$f") )); done
      free=$(df -B1 --output=avail "$dir" | tail -1)
      where="this Spark"
      run() { bash -c "$pack -v '$HF_CACHE:/hf:ro' -v '$dir:/out' --entrypoint tf-dsv41-engram-pack '$IMAGE' '$SNAP_C' /out $node $nodes"; }
    else
      have=$(worker "$r" "s=0; for f in $ENGRAM_PACKED_DIR/engram-l*-n${node}of${nodes}.tfe; do [[ -f \$f ]] && s=\$(( s + \$(stat -c %s \$f) )); done; echo \$s")
      free=$(worker "$r" "mkdir -p $ENGRAM_PACKED_DIR && df -B1 --output=avail $ENGRAM_PACKED_DIR | tail -1")
      where=$(worker_host "$r")
      run() { worker "$r" "$pack -v '$NFS_VOLUME:/hf:ro' -v $ENGRAM_PACKED_DIR:/out --entrypoint tf-dsv41-engram-pack '$IMAGE' '$SNAP_C' /out $node $nodes"; }
    fi
    if (( have >= pneed )); then log "Rank $r ($where): Engram's packed shards of node $node of $nodes on its disk"; continue; fi
    (( free + have > pneed + 10 * 1024 ** 3 )) || die "$where has $(( free / 1024 ** 3 )) GiB free: Engram's packed shards need $(( pneed / 1024 ** 3 )) GiB (ENGRAM_PACKED_DIR) and 10 GiB to spare"
    log "Rank $r ($where): Engram's packed shards of node $node of $nodes ($(( pneed / 1024 ** 3 )) GiB from the checkpoint, once)"
    run >/dev/null || die "packing Engram's shards on $where failed"
  done
fi
for i in 1 2 3; do
  [[ "$ENGRAM_PACKED" != 1 ]] || break
  have=$(worker "$i" "d=$WORKER_ENGRAM_DIR; s=0; for f in ${shards[*]}; do [[ -f \$d/\$f ]] && s=\$(( s + \$(stat -c %s \$d/\$f) )); done; echo \$s")
  if [[ "$have" == "$need" ]]; then log "Worker $i ($(worker_host "$i")): Engram tables on its disk"; continue; fi
  free=$(worker "$i" "mkdir -p $WORKER_ENGRAM_DIR && df -B1 --output=avail $WORKER_ENGRAM_DIR | tail -1")
  (( free + have > need + 10 * 1024 ** 3 )) || die "$(worker_host "$i") has $(( free / 1024 ** 3 )) GiB free: Engram's tables need $(( need / 1024 ** 3 )) GiB on its disk (WORKER_ENGRAM_DIR) and 10 GiB to spare"
  log "Worker $i ($(worker_host "$i")): copying Engram's tables to its disk ($(( need / 1024 ** 3 )) GiB over NFS, once)"
  worker "$i" "docker run --rm --network none --user \$(id -u):\$(id -g) -v '$NFS_VOLUME:/hf:ro' -v $WORKER_ENGRAM_DIR:/out --entrypoint bash '$BASE_IMAGE' -c 'for f in ${shards[*]}; do cp /hf/$SNAP_REL/\$f /out/\$f.part && mv /out/\$f.part /out/\$f || exit 1; done'" ||
    die "copying Engram's tables to $(worker_host "$i") failed"
done

# ---------------------------------------------------------------- 8. each worker's weight file on its disk
# start.sh mounts it when it matches this Spark's copy, so a start reads no weights over NFS (RANK_LOCAL=0: none)
IFS=, read -r -a ord <<<"$(node_order)"
for i in 1 2 3; do
  [[ "$RANK_LOCAL" == 1 ]] || break
  rf=$(rank_file "${ord[i]}"); [[ -n "$rf" ]] || die "no weight file of node ${ord[i]} in $DATA_DIR/rank-cache"
  name=${rf##*/}; size=$(stat -c %s "$rf")
  have=$(worker "$i" "stat -c %s $WORKER_RANK_DIR/$name 2>/dev/null || echo 0")
  if [[ "$have" == "$size" ]]; then log "Worker $i ($(worker_host "$i")): its weight file on its disk"; continue; fi
  free=$(worker "$i" "mkdir -p $WORKER_RANK_DIR && df -B1 --output=avail $WORKER_RANK_DIR | tail -1")
  if ! (( ${free:-0} > size + 10 * 1024 ** 3 )); then
    warn "$(worker_host "$i") has $(( ${free:-0} / 1024 ** 3 )) GiB free in WORKER_RANK_DIR, its weight file $(( size / 1024 ** 3 )) GiB and 10 to spare: rank $i keeps reading it over NFS"
    continue
  fi
  log "Worker $i ($(worker_host "$i")): copying its weight file to its disk ($(( size / 1024 ** 3 )) GiB over NFS, once)"
  worker "$i" "docker run --rm --network none --user \$(id -u):\$(id -g) -v '$NFS_VOLUME:/hf:ro' -v $WORKER_RANK_DIR:/out --entrypoint bash '$BASE_IMAGE' -c 'cp /hf/$DATA_SUB/rank-cache/$name /out/$name.part && mv /out/$name.part /out/$name'" ||
    die "copying the weight file to $(worker_host "$i") failed"
done

write_prepared_marker || die "cannot record the prepared state (the previous marker is left as it was): see the error above"
log "Done: all 4 Sparks are ready. Start the server with ./start.sh (port $PORT)."

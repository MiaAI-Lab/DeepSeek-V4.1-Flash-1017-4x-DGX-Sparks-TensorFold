<h1 align="center">DeepSeek-V4.1-Flash-1017 / 4x DGX Sparks / TensorFold</h1>

<p align="center">
  <sub>by <a href="https://x.com/MiaAI_lab">Mia's AI Lab</a></sub>
  <br><br>
  <a href="https://github.com/sponsors/MiaAI-Lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Sponsor%20me%20on%20GitHub-181717?style=for-the-badge&logo=githubsponsors&logoColor=white" alt="Sponsor me on GitHub" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
  <a href="https://x.com/MiaAI_lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow Mia on X" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
</p>

<img src=".github/pic.png" alt="DeepSeek V4.1 Flash on 4x DGX Sparks: 1M context, 16M KV pool, 65 tok/s prose, 16 concurrent sessions" width="100%" />

Serve DeepSeek's own **[DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)**
(FP8 / MXFP4, as published, no re-quantization) from four NVIDIA DGX Sparks (GB10, 128 GB each) over their
ConnectX-7 ports, with tensor parallelism over all four (TP4), through an OpenAI-compatible API: text,
**image and video input**, tool calling, thinking on or off, and **DSpark** speculative decoding from the checkpoint's
own draft blocks.

> [!TIP]
> **No switch needed.** This recipe runs the four Sparks either way: a plain cabled **switchless ring**
> (`./start-switchless.sh`, no RoCE switch -- the setup this README's own numbers are measured on) or a normal
> switched mesh (`./start.sh`). See [How it is split across the four Sparks](#how-it-is-split-across-the-four-sparks)
> for the difference.

> [!TIP]
> **The KV pool runs at its true ceiling: 16,777,216 positions total** (`POOL`, [Configuration](#configuration)) --
> `PARALLEL(16) x CONTEXT(1,048,576)`, so all 16 concurrent streams can each hold the model's full ~1M-token context
> at once, with nothing wasted past that. On this setup it leaves three of the four Sparks at ~15-18 GiB free and the
> fourth (which also runs other, unrelated workloads) at ~11 GiB -- all comfortable, no other tenant starved. Lower
> `POOL` if a Spark on your own setup runs other memory-heavy work too.

**Who wrote what.** The engine is [TensorFold](https://github.com/ashhart/TensorFold)'s native Zig engine, by Ash Hart
([ashhart](https://github.com/ashhart)) and the TensorFold contributors. The DeepSeek-V4.1-Flash family on it (the
four-node 2D split, Engram, DSpark drafting, the vision tower, the weight-file and token-map tools) and the kernel kit
were written by **[BertholomusAI (Albert Lee, @bertholomus)](https://github.com/bertholomus)**, in their fork
[bertholomus/TensorFold, branch `deepseek-v41-zig-tp4`](https://github.com/bertholomus/TensorFold/tree/deepseek-v41-zig-tp4)
at [`5940208`](https://github.com/bertholomus/TensorFold/commit/59402086381a42c74d53053443224eec2790b4c5) and their
recipe [deepseek-v4.1-tensorfold-tp4-4xgb10](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp4-4xgb10) at
[`6033ea4`](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp4-4xgb10/commit/6033ea48a27004b2715db82190e2a61f33411d92);
this recipe builds that engine and adds patches. The model and its reference inference code are
[DeepSeek](https://huggingface.co/deepseek-ai)'s. The chat template was written by Mia's AI Lab for
[`Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw`](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw). The
scripts are adapted from Mia's AI Lab's
[GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold),
with code by webzone (twin rails), [Alexbob0](https://github.com/Alexbob0) (workers by host name) and john (each
rank's rendezvous address). Full list: [Credits](#credits) and [`CREDITS.md`](CREDITS.md).

- Checkpoint: `deepseek-ai/DeepSeek-V4.1-Flash-1017` @ `1bae0e10`, ~510 GB, downloaded once on spark1; the other three
  Sparks read it, and their weight files, from spark1 over NFS
- API model id: `DeepSeek-V4.1-Flash-1017`, port 8888
- Engine: TensorFold native (Zig 0.17.0), BertholomusAI's `dsv41` family, plus [`zig/patches`](zig/patches):
  DeepSeek's native FP8 / MXFP4 weights, the ring's node order and links, video input, tool-call arguments
- One command on spark1: `./start.sh` (a switch) or `./start-switchless.sh` (a ring) sets up all four Sparks and
  starts the four ranks; `./stop.sh` stops them

**This recipe's own patches** ([`zig/patches`](zig/patches)) **are Mia's AI Lab's work, except `0008`, `0026` and `0027`**
(Hector, knapcio and uchkw, respectively) -- see [What the patches change](#what-the-patches-change) and [Credits](#credits) for who found or
shaped each specific fix along the way.

## Contents

- [Performance](#performance)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [ABLIT (opt-in)](#ablit-opt-in)
- [Images and video](#images-and-video)
- [DSpark drafting](#dspark-drafting)
- [NFS](#nfs)
- [Engram's packed rows (opt-in)](#engrams-packed-rows-opt-in)
- [How it is split across the four Sparks](#how-it-is-split-across-the-four-sparks)
- [Configuration](#configuration)
- [What the patches change](#what-the-patches-change)
- [Repository layout](#repository-layout)
- [License](#license)
- [Credits](#credits)

## Performance

Measured 2026-10-25 on four Sparks on a **switchless ring** (no RoCE switch; `./start-switchless.sh`) at the
defaults (1,048,576-token window, 16 streams, DSpark drafts, CUDA graphs, side stream and L2 prefetch on, the
measured `ROUND_MS` table and `DRAFT_MS`), from spark1 (median of three runs). "Agg" is the total tok/s across
every request at once, "/str" is one request's own share, TTFT is
time to first token. A switch removes the ring's one relayed hop between the two nodes with no direct cable; these
numbers are this fabric's, not a switch's ([more on the difference](#how-it-is-split-across-the-four-sparks)).

**Decode, prose (non-greedy, T = 0.7, thinking on):**

| Requests at once | Aggregate | Per request | TTFT |
| ---: | ---: | ---: | ---: |
| 1 | 65.8 tok/s | 65.8 tok/s | 186 ms |
| 2 | 96.5 tok/s | 48.9 tok/s | 304 ms |
| 4 | 130.5 tok/s | 34.1 tok/s | 499 ms |
| 8 | 165.3 tok/s | 21.7 tok/s | 901 ms |

**Decode, prose (non-greedy, T = 0.7, thinking on, each session starting at 128k context):**

| Requests at once | Aggregate | Per request | TTFT |
| ---: | ---: | ---: | ---: |
| 1 | 58.6 tok/s | 58.6 tok/s | 1.07 s |
| 2 | 83.1 tok/s | 44.3 tok/s | 1.61 s |
| 4 | 111.3 tok/s | 31.0 tok/s | 2.61 s |
| 8 | 132.3 tok/s | 19.2 tok/s | 4.67 s |

**Decode, code (greedy, T = 0, thinking off):**

| Requests at once | Aggregate | Per request | TTFT |
| ---: | ---: | ---: | ---: |
| 1 | 117.4 tok/s | 117.4 tok/s | 192 ms |
| 2 | 158.2 tok/s | 80.6 tok/s | 262 ms |
| 4 | 174.4 tok/s | 46.2 tok/s | 482 ms |
| 8 | 183.7 tok/s | 25.3 tok/s | 833 ms |

Three opt-in switches (off by default) add more, with the same replies: `TF_DS_EXPERT_EXIT=1` and `TF_DS_SAMPLE_CAND=1` together give level at one request,
+1.6 % to +2.8 % at 4 and +1.9 % to +8.2 % at 8 (measured with `TF_DS_PREFILL_YIELD=1` on as well, which costs nothing
here). `TF_DS_PREFILL_YIELD=1` (off by default) keeps the other requests going while a long prompt is read: with 15
requests decoding, a 25,000-token prompt stalled them ~7.5 s, ~1 s with it, for +35-41 % on that prompt's own time to
first token. Before `zig/patches/0017`, 13 or more drafting requests at once failed.

**Prefill, one request:**

| Context | Prompt tokens | TTFT | tok/s |
| ---: | ---: | ---: | ---: |
| 4k | 4,116 | 1.33 s | 3,497.9 |
| 8k | 8,210 | 2.28 s | 3,851.9 |
| 16k | 16,400 | 4.20 s | 4,051.9 |
| 32k | 32,787 | 8.10 s | 4,127.2 |
| 64k | 65,559 | 16.01 s | 4,134.1 |
| 128k | 131,093 | 32.45 s | 4,058.4 |
| 256k | 262,161 | 66.87 s | 3,929.4 |

**Needle in a haystack** ([`tools/needle_bench.py`](tools/needle_bench.py)): a random code hidden at 0/25/50/75/100%
depth, at nine context lengths from 4k up to 997,530 tokens (effectively this model's full 1,048,576-token window) --
**45 of 45 cells retrieved it correctly**, including the longest, deepest one (997k tokens, needle at 100% depth,
156 s). This recipe's prompt-sharding patches (quarter-HC, owner-row gather, indexer row-sharding) don't degrade
long-context retrieval at any tested depth.

Checked with [`tools/validate.py`](tools/validate.py) (all pass): text, thinking, a tool call and its result, an
image, a video, a 21,875-token prompt, 8 greedy replies sent at once equal to the same sent one at a time, and drafted
replies equal to undrafted ones. The native kernels are checked against DeepSeek's reference arithmetic on real
weights (`tools/dsv41-zig/native/check_native.py`).

## Requirements

- Four DGX Sparks, linked by their ConnectX-7 ports in one of two ways:
  - **on a switch** (every Spark on one RoCE network): `./start.sh`;
  - **as a ring without a switch**: spark1 - spark2 - spark3 - spark4 - spark1, one QSFP cable between neighbours
    (spark1 and spark3, spark2 and spark4 have none): `./start-switchless.sh`, with the workers listed in the ring's
    order.

  `scripts/nodes.sh` finds which one the Sparks are from their addresses; each script stops on the other kind and
  names the one to use.
- Docker with the NVIDIA runtime on every Spark, and key-based ssh from spark1 to the other three.
- On spark1: ~510 GB for the checkpoint and ~320 GB for the four weight files, under the Hugging Face cache
  (`HF_CACHE`, default `~/.cache/huggingface`), and an NFS export of that cache, read-only, to each worker's address
  on its link (see [NFS](#nfs)).
- About 100 GB of free memory on each Spark (the weights take ~80 GiB a node).

## Quick start

On spark1:

```bash
git clone https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold.git
```

```bash
cd DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold && cp scripts/local.sh.example scripts/local.sh
```

Set `WORKER`, `WORKER2` and `WORKER3` in `scripts/local.sh` to spark2, spark3 and spark4 (ssh targets; on a ring in
its order), and `NFS_PATH` if your export puts the cache somewhere other than `/huggingface`. Then, on a switch:

```bash
./start.sh
```

or, on a ring without a switch:

```bash
./start-switchless.sh
```

The first run builds the image, downloads the checkpoint, writes the four weight files (one per node, ~80 GB each) and
the workers' NFS volumes (`scripts/prepare.sh`), then starts the ranks and runs a smoke test. Later runs start in about
two and a half minutes. Try it:

```bash
curl -s localhost:8888/v1/chat/completions -H 'content-type: application/json' -d '{"model": "DeepSeek-V4.1-Flash-1017", "messages": [{"role": "user", "content": "Hello!"}], "chat_template_kwargs": {"enable_thinking": false}}'
```

`./start.sh restart` (or `./start-switchless.sh restart`) restarts with changed settings; `./stop.sh` stops all
four ranks.

To use Blackfrost-AI’s DERISKED FP8 checkpoint, set `DERISKED=1` in `scripts/local.sh`, or run
`DERISKED=1 ./start-switchless.sh restart`. Set `DERISKED=0` to return to the original checkpoint.
The first switch downloads and prepares separate weights; the API model id stays `DeepSeek-V4.1-Flash-1017`.

## ABLIT (opt-in)

`ABLIT=1` grafts [drowzeys's Keys abliteration overlay](https://huggingface.co/drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated)
onto the checkpoint before the rank caches are baked: a refusal-direction ablation of `attn.wo_b`, layers 10-35 only
(L0-9, L36-39, all MTP, vision, Engram, every expert, embed/head stay stock — see the overlay's own `ABLIT_META.json`
for the recipe). It is gated and free: accept its terms while logged in on Hugging Face, then `export HF_TOKEN=hf_...`
or `hf auth login` with that account, before:

```bash
ABLIT=1 ./start-switchless.sh restart
```

Set `ABLIT=0` to return to the stock checkpoint. The first switch downloads the overlay's sidecar (~1.1 GB) and
rebuilds all four weight files from it (the usual "~80 GB, a few minutes" rank-cache build, once).

**Why this needed its own mechanism, not the vendor's script.** The overlay is published and vendor-validated
against SGLang/vLLM; its own TensorFold-validated recipe needs a different, EXL3-quantized checkpoint, not the
native FP8/MXFP4 one this recipe serves. The vendor's apply script works by rewriting `model.safetensors.index.json`'s
`weight_map` -- a convention our engine never reads: `checkpoint.zig`'s `Shards.open` scans every `*.safetensors`
file in a directory by its own embedded header, keeping the last tensor name it sees in sorted-path order. So
`scripts/prepare.sh` instead builds a new checkpoint view with every stock file linked plus the overlay's sidecar
dropped in beside them -- `wo_b_l10_35_1017.safetensors` sorts after every `model-...-of-00048.safetensors` shard,
so its 26 layers' `attn.wo_b` win the lookup, and that view is what `tf-dsv41-rank-cache` bakes into the rank caches.

**Validated on this cluster**, not just the vendor's: a tensor-level check that the grafted `layers.N.attn.wo_b.weight`
in the built checkpoint view byte-matches the overlay's sidecar (not stock's); `tools/validate.py` all passed; a
`tools/needle_bench.py` spot check at 65k and 262k tokens, 25%/75% depth, all passed at the same speed as the stock
checkpoint's numbers above -- the graft doesn't cost long-context retrieval.

## Images and video

User turns and tool results take `image_url` and `video_url` parts as base64 `data:` URLs (remote URLs are off).

- **Images** go to the checkpoint's vision tower on rank 0 (PNG natively; other formats through Pillow).
- **Video** has no tower of its own in this model, so the server samples the clip into frames and sends them as
  images, each after a text line `Video frame at <t> s:`, in time order. Frames: `TF_VIDEO_FPS` a second (default 2),
  at most `TF_VIDEO_MAX_FRAMES` (default 32; a longer clip's frames are spread evenly over it), each scaled to at most
  `TF_VIDEO_MAX_SIDE` pixels a side (default 768). PyAV in the image decodes MP4, WebM, MOV and the other formats FFmpeg
  reads. Every frame costs prompt tokens like an image.

```json
{"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA..."}}
```

## DSpark drafting

The checkpoint carries DSpark draft blocks: after each decode step they propose a block of 5 tokens, which the next
step verifies in one pass. Verification is exact (a drafted reply equals the undrafted one), so drafts only change
speed. `DRAFTS=0` turns them off. On prose they give about 1.8 tokens a step.

## NFS

Every Spark reads the checkpoint, its weight file and the kernel kit from spark1's Hugging Face cache over NFS: spark1
holds the only copy. spark1 must export the cache read-only to each worker's address on its link (spark3, across the
ring, mounts through spark2's routed address). With an export whose root (`fsid=0`) holds the cache at `/huggingface`,
`NFS_PATH=/huggingface` (the default). `scripts/prepare.sh` makes a docker volume `dsv41-hf` on each worker, mounts it
and checks that every file a rank reads is there, at the right size.

## Engram's packed rows (opt-in)

Each decode round and prompt chunk reads Engram rows from two tables of ~384 million rows each (~203 GB). By default
every row costs two O_DIRECT reads (its FP8 values and its scales sit in separate tensors), the pairs of Sparks read
the same rows, nothing is cached, and the rows are decoded to bf16 on the CPU while the round waits.
`ENGRAM_PACKED=1` (patch 0003) changes where the rows come from, not what they are:

- `scripts/prepare.sh` writes packed shards once on every Spark's own disk with `tf-dsv41-engram-pack`: each row's
  values and scales side by side, so a row is one read.
- `ENGRAM_OWNED=1` (the default with packing): each Spark keeps only its quarter of the rows (~51 GB) and reads only the
  rows it owns; one byte-max exchange a layer fills in the rest, exact because one Spark holds each row and the others
  send zeros. `ENGRAM_OWNED=0` keeps the whole tables on every Spark (~203 GB each) and adds no exchange.
- A row cache in memory (`ENGRAM_CACHE_MIB`, 2048 by default): n-grams repeat, so many rows never reach the disk again.
- The FP8 rows go to the GPU as they are and a kernel decodes them, value for value as the CPU did.
- The fill's Engram read-ahead worker stays: with packing it stages each chunk's records a chunk ahead, in the same
  two slots a layer, instead of decoded rows.
- `ENGRAM_INSTREAM=1` (needs `ENGRAM_AIO=1`): a round's wait for its rows runs as a host step on the GPU stream, so the
  host does not stop between the round's graphs.

The rows a layer reads are the same bytes either way, so replies should not change. This path is new and unmeasured:
check `tools/validate.py` and replies against the default path before you rely on it, with `TF_DS_DECODE_DEC=0` while
the decode kernels' known issue stands, so a difference is the Engram path's. With packing on, the workers' old copies
of the tables (`WORKER_ENGRAM_DIR`) are no longer read and can be removed.

## How it is split across the four Sparks

BertholomusAI's 2D split: each Spark is a node `g = 2p + r` of a 2 x 2 grid, its row partner `g ^ 1` and column
partner `g ^ 2`. On a ring, `ORDER=auto` places the nodes 0, 1, 3, 2 around it, so every partner pair has a cable of
its own; the partners' exchanges run over two-rank NCCL communicators, one a cable (`TF_DS_LINK_GRAPHS`, with the graph
files `scripts/nodes.sh` writes), and the all-gathers go round the ring. Rank 0 (spark1) also serves the API, runs the
vision tower and the sampler.

## Configuration

Settings come from the environment, then `scripts/local.sh`, then `./.env`, then the defaults in
[`scripts/config.sh`](scripts/config.sh), which documents each one. The main ones:

| Setting | Default | |
| --- | --- | --- |
| `DERISKED` | 0 | set to 1 to load [Blackfrost-AI’s DERISKED FP8 checkpoint](https://huggingface.co/Blackfrost-AI/DeepSeek-V4.1-Flash-1017-DERISKED-FP8); explicit `MODEL_ID` / `MODEL_REVISION` override it |
| `ABLIT` | 0 | set to 1 for the Keys abliteration overlay ([above](#ablit-opt-in)); gated, needs `HF_TOKEN` |
| `CONTEXT` | 1048576 | a request's positions at most (the model's window) |
| `POOL` | 2097152 | the positions all requests share |
| `PARALLEL` | 16 | requests at once |
| `THINKING` | 1 | think first unless a request sends `enable_thinking: false` |
| `REASONING_EFFORT` | max | the effort when a request names none (`low` 50, `high` 75, `max` 100) |
| `TEMPERATURE` / `TOP_P` / `TOP_K` | 1.0 / 1.0 / 20 | the server's sampling defaults; a request's own win |
| `DRAFTS` | 1 | DSpark drafting |
| `GRAPHS` / `SIDE` / `PREFETCH` | 1 / 1 / 1 | CUDA graphs, the side stream, the paced L2 prefetch |
| `ENGRAM_PACKED` / `ENGRAM_OWNED` | 0 / 1 | Engram's rows from packed shards; each Spark its quarter of them ([above](#engrams-packed-rows-opt-in)) |
| `ENGRAM_CACHE_MIB` / `ENGRAM_INSTREAM` | 2048 / 0 | with packing: the row cache a Spark; the round's wait as a stream host step |
| `ROUND_MS` | measured table | a decode round's ms by its rows, the price the lane uses to choose drafts ([Performance](#performance)) |
| `DRAFT_MS` | 25 | the one-stream draft chooser's per-round constant (`TF_DS_DRAFT_MS`) |
| `SERVED_K` | solo | `solo`, `all` or `off`: where the per-stream draft chooser trims a round (`TF_DS_SAMPLE_CAND`, `TF_DS_PREPASS`, `TF_DS_EXPERT_EXIT`, `TF_DS_PREFILL_YIELD`, `TF_DS_DRAFT_ROW_MS` below are engine switches) |
| `PORT` | 8888 | |

## What the patches change

[`zig/patches`](zig/patches) apply to BertholomusAI's fork at `5940208`, in order. Flags are off by default unless
marked **on**; "on, this cluster" means `scripts/local.sh` turns it on here, not the shipped default. Full
measurements and rationale: [`CHANGELOG.md`](CHANGELOG.md); authorship: [`CREDITS.md`](CREDITS.md).

| Patch | Developer | Flag | What it does |
| --- | --- | --- | --- |
| `0001` | Mia's AI Lab | -- (on) | Reads DeepSeek's FP8/MXFP4 checkpoint as published, on new CUDA kernels; ring-without-switch node order and links |
| `0002` | Mia's AI Lab | -- (on) | Video input; tool-call arguments as JSON |
| `0003` | Mia's AI Lab | `TF_DS_ENGRAM_PACKED` | Engram's packed rows ([above](#engrams-packed-rows-opt-in)) |
| `0004` | Mia's AI Lab | -- (on) | Kernel loader falls back to a general Triton variant when none was captured |
| `0005` | Mia's AI Lab | -- (on) | A resumed chunk's positions copied to a 16-byte boundary |
| `0006` | Mia's AI Lab | -- (on) | `--reasoning-effort` accepts `max` |
| `0007` | Mia's AI Lab | -- (on) | Indexer row blocks aligned to 16 rows |
| `0008` | **Hector** | `TF_DS_PAIR_SUM` (on, this cluster) | Pair-sum gathers over the column link instead of a four-quarter all-gather |
| `0009` | Mia's AI Lab | `TF_DS_LOG_TIMING` | Logs the Engram worker's wait, by chunk and layer |
| `0010` | Mia's AI Lab | `TF_DS_PAIR_SUM_NEGZERO` | Hardens `0008`'s startup agreement; warms the link |
| `0011` | Mia's AI Lab | `TF_DS_DUMP_LAYER` | Tensor dump takes a layer list, plus the attention-mix gather |
| `0012` | Mia's AI Lab | -- | WP0: an offline replay tool proving WP4's kernels are row-independent |
| `0013` | Mia's AI Lab | -- (on) | WP2: the K-split projection GEMMs' tiling |
| `0014` | Mia's AI Lab | `TF_DS_IDX_RB` (on, this cluster) | WP6(a): the indexer's kit kernel at RB=2 |
| `0015` | Mia's AI Lab | `TF_DS_QUARTER_HC` (on, this cluster) | WP4: each node computes only its own quarter of the HC-mix |
| `0016` | Mia's AI Lab | `TF_DS_OWNER_GATHER` (on, this cluster) | WP5: the quarter-HC gather moves only each rank's owned quarter |
| `0017` | Mia's AI Lab | -- (on) | Decode launches run in 64-row slabs (13+ drafting streams used to fail) |
| `0018` | Mia's AI Lab | `TF_DS_DEC_PDL` / `TF_DS_RDMA_PDL` / `TF_DS_TRITON_PDL_ATTR` / `TF_DS_PDL` | Every PDL attribute off, for chasing the decode nondeterminism |
| `0019` | Mia's AI Lab | `TF_DS_GRAPH_FAIL_TEST` (test only) | CUDA graph capture agreed across every rank before use |
| `0020` | Mia's AI Lab | `TF_DS_PREPASS` | Streams join the drafter's pass only where it pays (measured slower; kept for later) |
| `0021` | Mia's AI Lab | `TF_DS_SAMPLE_CAND` (on, this cluster) | GPU top-k candidate sampling instead of a CPU scan |
| `0022` | Mia's AI Lab | `TF_DS_EXPERT_EXIT` (on, this cluster) | Decode expert kernels skip inactive rows early |
| `0023` | Mia's AI Lab | `TF_DS_PREFILL_YIELD` (on, this cluster) | Other requests get a round between a long prompt's prefill chunks |
| `0024` | Mia's AI Lab | `TF_DS_DRAFT_ROW_MS` | Draft rows priced at a measured constant instead of the round table (mixed; kept for later) |
| `0025` | Mia's AI Lab | `TF_DS_IDX_SHARD` (on, this cluster) | Indexer scores and picks only its own owned quarter of rows |
| `0026` | knapcio | -- (on) | RDMA queue pairs use the port's active MTU (capped at 4096) instead of a fixed 4096, so RDMA works on a switch whose Ethernet MTU is 1500 |
| `0027` | uchkw | -- (on with RDMA) | Parks idle RDMA proxies with a futex after all ranks quiesce; wakes before the next compute frame |

### Idle RDMA CPU use

Patch `0027` stops the three RDMA proxy threads from busy-polling while the engine has no live streams.
All ranks first synchronize their CUDA contexts and complete a READY/COMMIT handshake, then park the proxies
in a Linux futex wait. A compute frame wakes them before GPU work is enqueued. Active RDMA polling and numerical
kernels stay unchanged. All ranks must use the same image because the idle handshake adds a wire frame.

On one four-GB10 switchless-ring deployment (2026-10-26), Container idle CPU fell from about 300% per node
(three cores) to 0.21-0.32%; three idle/wake chat cycles passed. The deployment used a 2,097,152-position pool,
1,048,576-token context, 16 streams and the original 1017 weights. This verifies that configuration, not the
full feature matrix. A repeated A/B/A/B recheck with 60-second cgroup CPU windows confirmed the reduction
([measurements](docs/idle-cpu-validation.md)). Shared-circuit wall-power readings were noisy and incomplete; no isolated watts-saved
claim is made. Rebuild and replace all four images; weight conversion is not required for this patch.

## Repository layout

```
start.sh                  start all four ranks on a switch
start-switchless.sh       the same on a ring without a switch
stop.sh                   stop all four ranks
scripts/config.sh         settings and their defaults
scripts/nodes.sh          workers, links, ring detection, NCCL graph files, NFS volumes
scripts/prepare.sh        image, checkpoint, kit, weight files, served folder, NFS checks
scripts/banner.sh         the start banner
zig/Dockerfile            the engine image
zig/patches/              the engine patches
model/chat_template.jinja the chat template the server renders
```

## License

This recipe's own scripts, patches and documentation: Apache License 2.0 ([`LICENSE`](LICENSE)); [`NOTICE`](NOTICE)
lists the third-party work and its licenses. The model's weights are DeepSeek's, under the MIT license, and are not
part of this repository.

## Credits

- **[DeepSeek](https://huggingface.co/deepseek-ai)**: DeepSeek-V4.1-Flash-1017, its weights, its DSpark draft blocks
  and its reference inference code (MIT).
- **[TensorFold](https://github.com/ashhart/TensorFold)** by Ash Hart ([ashhart](https://github.com/ashhart)) and the
  TensorFold contributors (Apache-2.0): the native Zig engine and its OpenAI-compatible server.
- **[Mia's AI Lab](https://github.com/MiaAI-Lab/)**: this recipe; patches `0001`-`0007`, `0009`-`0025` in `zig/patches` (the checkpoint format on the
  native engine, the ring's node order and links, video input, tool-call arguments, kernel-variant fallbacks,
  alignment fixes, the prefill plan's WP0/WP1/WP2/WP4/WP5/WP6(a), the decode plan's G0.1-G0.3 and tracks A2/B/C/D/A3,
  and WP6(b)'s indexer row-sharding); and the chat template of
  [`Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw`](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw).
- **[BertholomusAI (Albert Lee, @bertholomus)](https://github.com/bertholomus)**: the DeepSeek-V4.1-Flash family on
  TensorFold's native engine
  ([bertholomus/TensorFold@`5940208`](https://github.com/bertholomus/TensorFold/commit/59402086381a42c74d53053443224eec2790b4c5)),
  the kernel kit
  ([deepseek-v4.1-tensorfold-tp4-4xgb10@`6033ea4`](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp4-4xgb10/commit/6033ea48a27004b2715db82190e2a61f33411d92))
  and the served settings this recipe starts with.
- **[Hector](https://github.com/hecisaza)**: patch `0008` (pair-sum gathers, `TF_DS_PAIR_SUM`) and the prefill measurements behind it.
- **[GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold)**
  contributors, whose code the scripts keep: webzone (both PCIe links of a QSFP port as rails,
  [`d5afeb0`](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/commit/d5afeb0)),
  [Alexbob0](https://github.com/Alexbob0) (workers by host name and `FABRIC_PEER`,
  [`5e1bf27`](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/commit/5e1bf27)), john (each
  rank's rendezvous at its link's address of the head,
  [`58eaed0`](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/commit/58eaed0)).
- [PyAV](https://github.com/PyAV-Org/PyAV) (BSD-3-Clause), [Zig](https://ziglang.org) (MIT), NVIDIA's PyTorch
  container, [huggingface_hub](https://github.com/huggingface/huggingface_hub) (Apache-2.0).

All the details: [`CREDITS.md`](CREDITS.md). "DeepSeek" belongs to DeepSeek; "DGX Spark" and "GB10" belong to NVIDIA.
This recipe is not affiliated with or endorsed by DeepSeek, NVIDIA, the TensorFold authors or BertholomusAI.

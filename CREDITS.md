# Credits

This repository is a recipe: scripts, a Dockerfile and engine patches that put other people's work together on four
DGX Sparks. Almost everything that makes it work was built by others. Its own work is licensed under the Apache License
2.0 ([`LICENSE`](LICENSE)); [`NOTICE`](NOTICE) carries the notices that go with it.

## Model

- **[DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)** by
  [DeepSeek](https://huggingface.co/deepseek-ai): the model, its weights (MIT, Copyright (c) 2023 DeepSeek) and its
  reference inference code. The weights are not part of this repository; `scripts/prepare.sh` downloads them from
  Hugging Face and serves them as published.
- The native FP8 / MXFP4 kernels in `zig/patches` compute what DeepSeek's reference inference code computes
  (`inference/model.py`'s `linear`, `Expert` and `MoE`; `inference/kernel.py`'s `act_quant`, `fp8_gemm` and `fp4_gemm`;
  `inference/convert.py`'s bf16 `wo_a`): the arithmetic rules were read there and written anew in CUDA; no DeepSeek
  code is included.
- **[Keys abliteration overlay for DeepSeek-V4.1-Flash-1017](https://huggingface.co/drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated)**
  by drowzeys: the `attn.wo_b` (layers 10-35) weights `ABLIT=1` grafts onto the checkpoint, gated (MIT). Not part of
  this repository; `scripts/prepare.sh` downloads the sidecar and builds the overlaid checkpoint view when set.

## Engine

- **[TensorFold](https://github.com/ashhart/TensorFold)** by Ash Hart ([ashhart](https://github.com/ashhart)) and the
  TensorFold contributors (Apache-2.0): the native Zig engine, its OpenAI-compatible server, its lane core, CUDA runtime,
  Triton AOT loader and EXL3 module.
- **The DeepSeek-V4.1-Flash family on TensorFold's native engine** was written by
  [BertholomusAI (Albert Lee, @bertholomus)](https://github.com/bertholomus): the `dsv41` family, its exact
  four-node 2D split with the balanced experts split, its Engram reader, DSpark drafting, vision tower and the tools
  that write each node's weight file and Engram's token map, in their fork
  [bertholomus/TensorFold, branch `deepseek-v41-zig-tp4`](https://github.com/bertholomus/TensorFold/tree/deepseek-v41-zig-tp4)
  at commit [`5940208`](https://github.com/bertholomus/TensorFold/commit/59402086381a42c74d53053443224eec2790b4c5)
  (Apache-2.0). This recipe builds that commit and applies its patches on top.
- **The kernel kit** (the family's Triton kernels as captured binaries, Engram's constants, the RoPE table generator,
  the vision tower's attention kernel) is the `kit/` of BertholomusAI's four-node recipe
  [bertholomus/deepseek-v4.1-tensorfold-tp4-4xgb10](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp4-4xgb10)
  at v0.1.1 ([`6033ea4`](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp4-4xgb10/commit/6033ea48a27004b2715db82190e2a61f33411d92)),
  authored by BertholomusAI (Apache-2.0); `scripts/prepare.sh` fetches it at that commit and checks it against its
  MANIFEST. The kit's own licenses (`kit/KIT-LICENSES.md`) cover the parts it carries: Triton (MIT), PyTorch's attention
  kernel (BSD-3-Clause, built on NVIDIA CUTLASS, BSD-3-Clause) and the EXL3 format of
  [ExLlamaV3](https://github.com/turboderp-org/exllamav3) by turboderp (MIT).
- The served settings `scripts/config.sh` starts with (`TF_DS_2D_GU=parity`, 48-row rounds, CUDA graphs, the side
  stream, the paced L2 prefetch, Engram by AIO, the lane core's round costs `ROUND_MS`) are BertholomusAI's, from their
  recipe's `config.env.example` and README ("The served settings").

## This recipe's patches (`zig/patches`)

**Every patch is Mia's AI Lab's own work, except `0008`, which is Hector's.** At a glance:

| Patches | Author | What |
| --- | --- | --- |
| `0001`-`0007` | Mia's AI Lab | the checkpoint format on the native engine, the ring's node order and links, video input, tool-call arguments, kernel-variant fallbacks, alignment fixes |
| `0008` | **Hector** | pair-sum gathers (`TF_DS_PAIR_SUM`) |
| `0009`-`0016` | Mia's AI Lab | the prefill plan: WP0 tooling and timing, WP1 hardening, WP2 projection-tile tuning, WP4 quarter-HC, WP5 owner-row transport, WP6(a) indexer RB=2 |
| `0017`-`0024` | Mia's AI Lab | the decode plan: G0.1-G0.3, tracks A2/B/C/D/A3 |
| `0025` | Mia's AI Lab | WP6(b), the indexer's own row-sharding |

The per-patch notes below also credit everyone who helped find or shape a specific fix along the way -- Hector's
reviewer, Codex and Grok's live-bug diagnosis and design reviews, an outside reviewer's observation -- without
changing who wrote and shipped the patch itself.

Written for this recipe by Mia's AI Lab on BertholomusAI's engine (`0001` to `0007`):

- DeepSeek's own checkpoint format on the native engine: FP8 E4M3 linears with UE8M0 scales on 32 x 32 blocks, MXFP4
  routed experts, bf16 `wo_a`, router, head, compressor and indexer projections (`native.cu`, `mx.zig`, the loader and
  the call sites), with checks against DeepSeek's reference arithmetic on real 1017 weights
  (`tools/dsv41-zig/native/check_native.py`).
- `TF_DS_2D_ORDER`: the 2D split's node each NCCL rank serves, so that four Sparks cabled as a ring without a switch put
  every row and column partner on a cable of its own; and `TF_DS_LINK_GRAPHS`: the partners' exchanges over two-rank
  NCCL communicators, one a cable, each with its own graph file; and `TF_TP_LISTEN`: rank 0 listens for each worker
  at the address of that worker's own cable, the address john's `start.sh` change gives it.
- The server (`0002`): video input (`video_url` parts become timestamped frames for the vision tower, decoded by PyAV),
  and tool-call arguments sent as JSON strings parsed into objects before the chat template renders them.

Written by [Hector](https://github.com/hecisaza), on the recipe at `32925ca`:

- `0008` (pair-sum gathers, `TF_DS_PAIR_SUM`): a prompt chunk's two-partial fp32 sums over the cable between the TP2
  partners, swapped with the other pair, with the ranks' agreement on the setting at start. Hector also measured the
  prefill profile and the options that led to it; the patch was reviewed twice with Codex.

Written for this recipe by Mia's AI Lab on BertholomusAI's engine (`0009` to `0016`), from the prefill plan's work
packages. `0009`-`0011` also close the two Codex reviews of patch `0008` in Hector's `ds41-prefill-findings.zip`:

- `0009` (`TF_DS_LOG_TIMING`): the Engram worker's wait inside `prompt.engramWorkerRows`, printed per chunk and
  Engram layer -- the gap existing counters miss, since it precedes `engramApply`.
- `0010` (pair-sum hardening): `agreePairSum`'s startup exchange now carries every rank's node order
  (`TF_DS_2D_ORDER`) and gather widths, not just its setting and a locally-derived "my links look fine" bit, and
  refuses to start on any difference -- closing the counterexample in Hector's reviewer's first open item (three
  ranks on one role permutation and one on another, each passing a same-setting-only check while addressing
  different partners under the same role numbers); warms one `ncclAllReduce` per link at startup, alongside the
  existing all-gather warm-up, so `TF_DS_PAIR_SUM`'s first real gather doesn't pay NCCL's first-collective
  connection cost (the reviewer's item 4); and `TF_DS_PAIR_SUM_NEGZERO`, the reviewer's own suggested cheap
  experiment (item 2): fills the gather's zero plane with negative instead of positive zero.
- `0011` (`TF_DS_DUMP_LAYER` as a comma list, and the attention-mix gather `ga` alongside the existing `gm`): tensor
  dumps at several layers in one run, for comparing what `TF_DS_PAIR_SUM` changes against a flag-off capture of the
  same prompt.
- `0012` (`tf-dsv41-hc-replay`, a new standalone executable, no change to anything served): calls the real
  AOT-compiled HC-mix kernels (`_hc_mix_pf`, `_hc_finish`) on bounded random data, once as a 2048-row launch and
  once as four 512-row launches, and compares the posted stream, coefficients and normed output bit for bit --
  the plan's WP0 4.3b, confirming the kernels WP4 (quarter-HC) would rely on are row-independent.
- `0013` (WP2, the K-split projection GEMMs' tiling): Hector's measured `PF_GEMM_FOLD` tuning --
  4 K-blocks a stage, 3 stages, instead of 2 and 4 -- built into the
  image for wq_a / wkv / wq_b and every other dense FP8 GEMM that shares the kernel. Checked bit-identical to the
  previous tiling on real weights (`check_native.py`, every row count and split tested, against both the tile
  kernel and an fp64 reference) before landing.
- `0014` (WP6(a), the indexer's prompt-chunk scoring kernel): `TF_DS_IDX_RB`, which lets `_index_score_rows` use the
  kit's own RB=2 capture (BertholomusAI's kit again -- a second, already-compiled variant of the same kernel,
  one program a 2 rows instead of 1, so the 64-key tile it dequantizes is shared instead of redone per row) in
  place of the RB=1 the engine always used. Checked bit-identical to RB=1 (`tf-dsv41-idx-replay`, a new
  standalone executable: real AOT kernel, both of `prompt.zig`'s call shapes, even and odd row counts) before
  adding the flag.
- `0015` (WP4, quarter-HC): `TF_DS_QUARTER_HC`, each of the four nodes computing the HC-mix step only for its own
  512-row quarter of a full 2048-row chunk and all-gathering the result, instead of every node redundantly
  computing the full chunk. The row-sharded design, the kernel-buffer-stride fixes it needed (the gathered-plane
  and Sinkhorn-scratch buffers both have to be freshly, tightly sized per quarter rather than sliced from the
  full chunk's) and the live validation are Mia's AI Lab's. **The defect that made three independent fix attempts
  fail live before this patch was correct -- the layer-20 boundary reading stale `post`/`comb` coefficients for
  three of the four quarters -- was diagnosed by Codex (OpenAI) and Grok (xAI), consulted as an independent
  second-opinion council on the exact same bug after three self-diagnosed hypotheses (a kernel stride bug, a
  missing NCCL warm-up, a missing synchronize) had each been fixed or tested and none had resolved the live
  failure.** Both, working independently from the same brief and the same patched source tree, converged on the
  identical root cause and the identical fix (gather `post` and `comb` at the boundary, not just `h` and `pre`)
  without seeing each other's answer; their diagnosis was then proven offline (a new differential test,
  `tf-dsv41-wp4-replay`, built specifically to simulate the failure and confirm the fix, on real AOT kernels, no
  live cluster) before the fix was ever tried live, including the three failed attempts before it.
- `0016` (WP5, owner-row transport): `TF_DS_OWNER_GATHER`, shrinking a row-sharded quarter-HC layer's gather to
  the one quarter each rank actually owns, moved over the existing two-rank column link (`pairSum`'s own
  pattern) instead of a new collective. **Before any code was written, Codex (OpenAI) and Grok (xAI) were
  consulted again, this time as an independent design-review council** on whether WP5 was still worth
  attempting now that quarter-HC changed its rationale, and on the plan's own point-to-point send/recv-with-
  relay design. Both independently recommended against that design (a likely NCCL group-ordering hazard across
  the diagonal relay) and independently proposed the same safer alternative -- reuse the existing link
  collectives instead of a new primitive -- which this patch implements. The implementation, the offline
  host-side schedule differential that proved it before any live attempt, and the live validation are Mia's AI
  Lab's.

Written for this recipe by Mia's AI Lab on BertholomusAI's engine (`0017` to `0024`), from the decode plan (four
council rounds with Codex (OpenAI) and Grok (xAI), then run end to end on the four Sparks):

- `0017` (G0.1, the drafter's width): past 64 rows the dense decode launches run in 64-row slabs, so 13 to 16 streams
  drafting in one round are served. **The failure it fixes -- every request on the server failing with
  `NotADecodeWindow` as soon as 13 or more streams draft in one round, because the drafter's pass of five rows a stream
  reaches dense decode launches that refuse more than 64 rows -- was found by Codex (OpenAI) in the decode council's
  first round, from the source, before any test had run more than eight streams; it was then reproduced on this
  cluster (0 of 13 and 0 of 16 requests served) and fixed here.
- `0018` (G0.2, the decode-PDL controls `TF_DS_DEC_PDL`, `TF_DS_RDMA_PDL`, `TF_DS_TRITON_PDL_ATTR`, `TF_DS_PDL`): the
  four switches the plan specified for the open nondeterminism; the Triton control's design (a launch-attribute
  override that keeps the kit's `PDL=1` cubins) is Codex's, from the council's round 3.
- `0019` (G0.3, graph captures agreed on every rank): addresses the incident Hector reported
  (`cuGraphInstantiateWithFlags` failing on one rank and the other ranks hanging).
- `0020` (A2, `TF_DS_PREPASS`, opt-in) and `0021` (Track B, `TF_DS_SAMPLE_CAND`, opt-in): measured and left off by
  default.
- `0022` (Track C, `TF_DS_EXPERT_EXIT`), `0023` (Track D, `TF_DS_PREFILL_YIELD`) and `0024` (A3, `TF_DS_DRAFT_ROW_MS`),
  all opt-in, written by Mia's AI Lab the same way; Codex's reviews found the failure and cancellation paths fixed in
  `0023` (an interleaved round's failure failing every live request, cancels taken between rounds, a cancel race
  closed under one lock) and the cost base fixed in `0024`.
- Codex (OpenAI) reviewed every one of these patches read-only before it was built; its findings shaped `0019`'s
  agreement protocol and `0020`'s cost model, and it found the checker defect fixed in `0021`'s harness.

Back to the prefill plan, written by Mia's AI Lab:

- `0025` (WP6(b), the indexer's own row-sharding): `TF_DS_IDX_SHARD`, each node scoring and picking only its own
  owned quarter of a full chunk's rows instead of every node redundantly running the indexer over all of them,
  then all-gathering the int64 picks back to full width -- the plan's own WP6(b), left unattempted until an
  independent third-party review of this session's prefill work (not Codex or Grok's council; a separate
  outside review of the shipped state) observed that the indexer was still repeated on every node, pointing
  straight at this gap. Checked bit-identical to the full, unsharded computation offline first
  (`tf-dsv41-wp6b-idx-shard-replay`: the real AOT scoring kernel, a full 2048-row launch against four 512-row
  launches at row offsets into the same buffers, RB=1 and RB=2, keyed and plain, 5 seeds) before any live attempt.

## Chat template

- **`model/chat_template.jinja`** was written by Mia's AI Lab ([Mia-AiLab](https://huggingface.co/Mia-AiLab)) for
  [`Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw`](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw):
  a Jinja port of the checkpoint's own `encoding/encoding.py` (DeepSeek, MIT). The file here is that repository's,
  unchanged.

## Scripts

- The scripts' structure, the layered settings (`scripts/config.sh`), the node discovery, ring detection, NCCL graph
  files and NFS volumes (`scripts/nodes.sh`), `stop.sh`, the start banner and the image's build pattern come from
  Mia's AI Lab's [GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold)
  (branch `zig-engine`), with code those contributors authored there:
  - **webzone**: both PCIe links of a cabled QSFP port as rails (the DGX Spark "twins") in `scripts/nodes.sh`
    ([`d5afeb0`](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/commit/d5afeb0));
  - **[Alexbob0](https://github.com/Alexbob0)**: workers given by host name and `FABRIC_PEER` resolution in
    `scripts/nodes.sh` ([`5e1bf27`](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/commit/5e1bf27));
  - **john**: each rank given its link's address of the head as its rendezvous (`LINK_HEAD_ADDR`), kept in
    `start.sh` ([`58eaed0`](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/commit/58eaed0)).

## Tools and platform

- [Zig](https://ziglang.org) 0.17.0 (MIT), the compiler of the engine.
- NVIDIA's PyTorch container `nvcr.io/nvidia/pytorch:26.07-py3`, the image's base: its CUDA toolchain, NCCL, cuBLAS,
  libibverbs, and the Python and Pillow the engine calls for picture formats other than PNG (NVIDIA's container license
  and the components' own).
- [PyAV](https://github.com/PyAV-Org/PyAV) 18.1.0 by the PyAV contributors (BSD-3-Clause; its wheel's FFmpeg
  libraries are LGPL-3.0-or-later), installed into the image for video input.
- [huggingface_hub](https://github.com/huggingface/huggingface_hub) (Apache-2.0), the `hf` download.
- [sparkDash](https://github.com/MiaAI-Lab/sparkDash) by Mia's AI Lab, the benchmark client of the performance numbers.

"DeepSeek" belongs to DeepSeek; "DGX Spark" and "GB10" belong to NVIDIA. This recipe is not affiliated with or endorsed
by DeepSeek, NVIDIA, the TensorFold authors or BertholomusAI.

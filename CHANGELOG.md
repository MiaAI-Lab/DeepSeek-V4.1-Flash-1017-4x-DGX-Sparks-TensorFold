# Changelog

- **2026-10-26** -- `0026` (knapcio): the RDMA rings' queue pairs take the port's active MTU (`ibv_query_port`,
  capped at 4096) as their path MTU instead of a fixed 4096. On a switched fabric whose Ethernet MTU is 1500 the RoCE
  active MTU is 1024, and with the fixed 4096 every request failed (`RDMA=1`); with the port's MTU it serves. On a
  fabric whose active MTU is 4096 the value is the same as before.
- **2026-10-25** -- `ABLIT` (new, off by default): grafts drowzeys's Keys abliteration overlay (`attn.wo_b`,
  layers 10-35) onto the checkpoint before the rank caches are baked, applied via a new checkpoint view rather
  than the vendor's own index-remap script (our engine never reads `model.safetensors.index.json`). Live-validated:
  tensor-level proof the grafted bytes match the sidecar, `validate.py` all passed, needle test passed at 65k/262k.
- **2026-10-25** -- `tools/needle_bench.py` (new): a needle-in-a-haystack retrieval sweep, 4k to ~997k tokens x
  0/25/50/75/100% depth. 45 of 45 cells passed.
- **2026-10-25** -- WP6(b), `TF_DS_IDX_SHARD` (`0025`): the indexer scores and picks only its own owned quarter of a
  chunk's rows, instead of every node redundantly scoring all of them. On by default (`scripts/local.sh`).
- **2026-10-25** -- resumable prefill, expert early exit, draft-row costs (`0022`-`0024`, opt-in): `TF_DS_PREFILL_YIELD`
  and `TF_DS_EXPERT_EXIT` on (this cluster); `TF_DS_DRAFT_ROW_MS` measured mixed, kept off.
- **2026-10-25** -- the decode plan's Gate 0 and draft economics (`0017`-`0021`, `ROUND_MS`, `DRAFT_MS`): 64-row
  drafting slabs fix 13+ concurrent drafting streams failing outright; CUDA graph capture agreed across every rank;
  PDL controls (off); `ROUND_MS`/`DRAFT_MS` set from measurement; `TF_DS_SAMPLE_CAND` opt-in.
- **2026-10-24** -- WP5, `TF_DS_OWNER_GATHER` (`0016`): the quarter-HC gather moves only each rank's owned quarter
  over the column link, instead of the full chunk. On by default.
- **2026-10-24** -- WP4, `TF_DS_QUARTER_HC` (`0015`): each node computes the HC-mix step for only its own quarter
  of a chunk, then all-gathers the result. On by default.
- **2026-10-24** -- WP6(a), `TF_DS_IDX_RB` (`0014`): the indexer's prompt-chunk scoring kernel at the kit's own RB=2
  capture instead of RB=1. Off by default (opt-in).
- **2026-10-24** -- WP2, projection-tile tuning (`0013`, Hector's measured tuning): a better tile config for the
  K-split dense FP8 GEMMs. On by default.
- **2026-10-24** -- WP0 4.3b (`0012`): an offline replay tool proves the HC-mix kernels are row-independent, the
  entry gate WP4 needed.
- **2026-10-24** -- pair-sum hardening (`0009`-`0011`): `TF_DS_PAIR_SUM`'s startup agreement hardened, Engram
  worker-wait timing, a multi-layer tensor dump.
- **2026-10-23** -- guarded live-trial tooling for the prefill work (removed later, once the work was done).
- **2026-10-23** -- pair-sum gathers, opt-in (`0008`, Hector): sums each column's two TP2 partials over the cable
  between the partners and swaps with the other pair, instead of a four-quarter all-gather.
- **2026-10-18** -- reasoning effort `max` by default; aligned indexer row blocks (`0006`, `0007`).
- **2026-10-18** -- a resumed chunk's positions copied to a 16-byte boundary (`0005`).
- **2026-10-18** -- a general Triton variant when no exact one was captured (`0004`).
- Engram's packed rows, opt-in (`0003`): off by default, not yet measured.
- **2026-10-18** -- the four speed agents' work merged: native FP8/MXFP4 kernels, the ring's node order and links,
  video input, tool-call arguments as objects.
- **v0.1.0 (unreleased)** -- first version: DeepSeek-V4.1-Flash-1017 on four DGX Sparks, TP4, on BertholomusAI's
  DeepSeek-V4.1-Flash family of TensorFold's native engine.

## Native server update — 2026-10-28

- `0028`: TensorFold native SSE keepalives, enabled every 15 seconds during silent streaming generations.
  Chat/Responses use comments; Messages uses ping events. `TF_SSE_KEEPALIVE_S=0` disables them.
- `0029`: drain the golden test server stdout so a full log pipe cannot stall the regression suite.
  Both fixes are upstream backports; authorship and source commits are recorded in `CREDITS.md`.

# Changelog

- **2026-10-28** -- `model/chat_template.jinja` writes each piece out as it is made instead of appending to one string
  with `{% set ns.out = ns.out + ... %}` (hashd1ve's fix from the two-Spark recipe, its issue #1 and `fd2cdb2`, carried
  over byte for byte by manuel1618 in #18, for #17). The engine's Jinja copies the whole string on every `+` and keeps
  each copy until the render ends, and every request renders twice, so rank 0's host memory grew with messages x prompt
  length: in the engine's own template module one render of a 1,500-turn conversation (1.6 MB of prompt) peaked at 24.8
  GB, now 0.4 GB. Prompts are unchanged: 14,688 conversations render byte-identically before and after, in the engine's
  template module and in jinja2. On this cluster: `/tokenize` of a 500-turn agent conversation (108k tokens) gave the
  same ids with both templates while rank 0's memory rose 1,967 MiB before and 110 MiB after (200 turns: 349 and 36
  MiB); `tools/validate.py` passed with its 8 greedy replies equal to the saved ones, and 5,000 random conversations
  rendered identically in jinja2.
- **2026-10-28** -- `0035` (Mia's AI Lab), the ranks' startup rendezvous
  ([#3](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/issues/3)). Rank 0 used to open
  its ports for workers 1, 2 and 3 one after another, each only once the worker before it had connected, and only
  after its own weights had loaded; a worker retries its connect for 300 s after loading its own weights. So a worker
  that loaded more than 300 s before the ones ahead of it exited with `error: ConnectFailed` (in #3: rank 3 loaded in
  113 s, ranks 1 and 2 in ~843 s over NFS). Rank 0 now opens all three ports when it starts, before its weights load,
  and a worker that connects early waits in its port's backlog. `TF_DS_CONNECT_S` sets a worker's retry window (300 s
  by default; it now covers only rank 0's own start), and a worker that gives up logs the address and port it tried.
  Diagnosed by shanedertrain in #3, reported by tlacerdap. Tested on this cluster with rank 0 and rank 3 started
  together and ranks 1 and 2 nine minutes later: before `0035` rank 3 exited with `ConnectFailed` 427 s in; with it,
  rank 3's connection waited on rank 0's port and the server answered 696 s in.
- **2026-10-28** -- seven pull requests merged (#2, #4, #5, #8, #10, #13, #15), tested together on this cluster. Three
  of them had named their patch `0027`: #10's keeps the number, #8's is `0033` here and #2's `0034`. Against the image
  before them (`0001`-`0025`), on four GB10s with this cluster's switches: `tools/validate.py` passed and its 8 greedy
  replies were equal; a 131k-token prompt read cold in 34.7 s as before; a follow-up resumed from the previous turn's
  state gave the same tokens as the same prompt read cold (8 of 8); a 35-image prompt resent read in 1.0 s instead of
  20.1 s, its tokens equal; a 12-screenshot history resumed on every turn, and a changed or swapped picture resumed none
  of what followed it; a 9-token request arriving 1 s into a cold 50,939-token prompt had its first token in 0.3 s; idle
  CPU fell from ~310 % to 0.15-0.35 % a rank.
- **2026-10-28** -- `0032` (Mia's AI Lab): `0031`'s bounded admission only with `TF_DS_PREFILL_YIELD=1` (on, this
  cluster). Without the switch a prompt is read whole, as before `0031`, and its Engram read-ahead keeps running
  between its chunks.
- **2026-10-28** -- `0030`, `0031` (Mauvis Ledford): `0030` skips vision work covered by a restored image-keyed prefix;
  `0031` admits and advances requests between unchanged prompt chunks. Partially filled streams keep separate progress
  and image rows on the head, and all TP ranks select the same slot when resuming. No arithmetic or checkpoint
  settings change.
- **2026-10-28** -- `0028`, `0029` (upstream TensorFold, backported by Mauvis Ledford): native SSE keepalives, every 15
  seconds during silent streaming generations (Chat/Responses use comments; Messages uses ping events;
  `TF_SSE_KEEPALIVE_S=0` disables them), and the golden test server's stdout drained so a full log pipe cannot stall the
  regression suite. Authorship and source commits are recorded in `CREDITS.md`.
- **2026-10-27** -- `0033` (manuel1618, its PR's `0027`): `/metrics` publishes `prompt_tokens_cached_total`, the prompt
  tokens of finished requests restored from kept prompt states, and repeats the split under vLLM's
  `prefix_cache_hits_total` / `prefix_cache_queries_total`. Before, `/metrics` had only `prompt_tokens_total`, which
  counts cached and computed prompt tokens alike, so a dashboard could not tell them apart (a 93 %-cached agent workload
  read as all computed). The per-request `usage.prompt_tokens_details.cached_tokens` is unchanged. Backport of
  BertholomusAI's `ce0d9053` to this recipe's engine pin, plus the one line that turns the series on for the
  DeepSeek-V4.1 engine.
- **2026-10-27** -- `0027` (Thomas Kempka), the image-keyed prompt cache: a kept prompt state is keyed by the request's
  images as well as its tokens, so a chat whose history holds a picture (a screenshot in a tool result, say) resumes
  its prefill instead of re-running it from 0 on every turn. Measured on 4x GB10 TP4 (16 streams, 2.8k tok/s prefill):
  one 150k-token prompt replayed in 0.64 s (82x) without an image and 53.6 s (1.0x) with one before this patch; both
  replay now. Text-only requests are unchanged (the key is 0 for every prefix, so the tokens alone decide), a state
  kept for a picture is never learned to disk, and it is not resumed by a prompt whose prefix holds no image.
- **2026-10-26** -- `0034` (uchkw, its PR's `0027`): park the three RDMA proxy threads when no streams are live.
  Synchronize CUDA on every rank, agree READY/COMMIT, then block in futex waits; wake before compute and before
  stop/join. Idle followers block on the control link instead of spinning. A four-GB10 ring measured about 300% CPU per
  node before and 0.21-0.32% after; three idle/wake chat cycles passed. Numerical kernels and weights are unchanged.
  All ranks must run the updated image (new idle control frame). Ported onto `0031` (its frame value is 11).
- **2026-10-26** -- (SvangenStudios) `stop.sh` exits 1 when a rank may still run (a rank it could not stop, or a Spark
  whose containers it could not list); `DRY_RUN=1` writes no NCCL graph files; the prepared marker also covers
  `HF_CACHE`, the NFS settings and the chat template, and a state that cannot be computed never reads as ready. Offline
  regression tests in `tests/test_scripts.py` (standard library only, no Spark).
- **2026-10-26** -- (SvangenStudios) `tools/validate.py`, `tools/needle_bench.py`: a check passes only on a real
  completion. An HTTP error, an `error` object, a reply that is not the assistant's, a malformed tool call or a short or
  empty `--compare` reference now fails instead of passing; bad arguments stop with a message. Offline regression tests
  in `tests/test_validation_tools.py` (standard library only, no GPU).
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

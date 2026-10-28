# Prefill latency validation — October 28, 2026

Validated on four DGX Sparks (GB10, 128 GB each), TP4 over a 200 Gb/s ring. Native FP8/MXFP4 weights, 16 streams,
1,048,576-token context and 2,097,152-position shared pool were retained. No model arithmetic, kernels, sampling
settings or draft-confidence calibration changed.

- TensorFold fork: `59402086381a42c74d53053443224eec2790b4c5`, with the recipe patches.
- Model revision: `1bae0e1088993b9b82de08f59de5be5a34e42170`.
- Kernel kit: `6033ea48a27004b2715db82190e2a61f33411d92`.
- Pinned Zig 0.17.0; native CUDA server built with the existing ReleaseSafe server settings.
- Image-keyed cache from Thomas Kempka's PR #10 was present on both sides. Native SSE keepalives and
  `TF_DS_PREFILL_YIELD=1` were also present on both sides.
- Native executable SHA-256: `84b2d2be6d59e989763d953b3517c25f5d5538b3144a4a6455bf3d7a74945d5c`.

## Observed causes and changes

The previous yield hook decoded existing streams between text chunks, but request admission still called a whole
prefill. A regression test submits a short request during the first of eight long-prompt chunks; it failed before
the host change and passes afterward, with both outputs equal to standalone reference streams.

Patch `0031` adds an optional bounded-prefill backend callback, forwards it through the CUDA context wrapper, and
rotates admitted prompts one existing chunk at a time. Each prompt retains its slot, extent, sequence, progress and
image rows. A new TP control frame selects the same resumed slot on every rank. The cache restore is reported before
another prompt can evict its entry. Cancellation and pool-full retries account for partially filled and unstarted
admitted requests. Failed image setup or slot selection invalidates the selected-slot marker before another prompt resumes.

A live request with 35 retained images spent 15.128 seconds before its first text chunk; restoring the snapshot itself
took less than one millisecond. The cached cut already covered all those image spans. The local transcript export
contained only two of the retained images, so the initial two-image reproduction understated this workload.

Patch `0030` skips vision work only when a usable, image-keyed snapshot covers the entire image span. Placeholder
masking is retained, and an image crossing the cached cut is encoded as before.

## Four-node inference results

Greedy sampling, seed 42, thinking disabled. This is an inference test, with no generated video or render duration.
The image fixture alternates two original JPEGs (1344×768 and 1320×502); the image contents are private and are not
included in this repository. The public validation tool accepts operator-supplied originals.

| Test | Before | After |
|---|---:|---:|
| Late 9-token request, first token, arriving 1 second after a cold 50,943-token prompt | 15.378 s | 0.680 s |
| Cold 50,943-token prompt, first token | 16.240 s | 16.677 s |
| 35-image / 48,007-token replay, prefill time, 47,104 tokens reused | 13.666 s | 0.516 s |
| Same replay overlapping a 768-token decoder, prefill time | 13.592 s | 0.537 s |
| Decoder's largest output-event gap during that test | 13.789 s | 0.635 s |
| Cold 35-image request, prefill time | 28.813 s | 29.145 s |

Every recorded output token ID matched before/after and solo/replay/overlap. The 35-image request generated 105 tokens;
the integer decoder generated 768. The late short and long requests each generated two tokens. The short request
finished before the long prompt reached its first token.

The original author image canary, changed-image cache isolation, and draft/plain equality passed. A further image
state test used 2,446-token and 410-token prompts: fresh and concurrent/resumed outputs matched exactly, including a
2,048-token cached cut through the larger prompt. Its first harness assertion incorrectly assumed that a prompt below
4,096 tokens could not be cached; the DeepSeek backend keeps its own chunk-end checkpoint. The raw result and failed
assertion were retained, and revalidation checked the actual cache cut and every token ID.

Baseline runs shared the server with live traffic; candidate acceptance tests ran while external traffic was drained.
These receipts establish correctness and removal of the observed long stalls; they are not an isolated throughput A/B.
After reopening traffic, real long-context cached continuations completed prefill in approximately 0.6–1.6 seconds.
Cold vision processing still precedes the first text chunk. This work does not establish a fix for the separate
rank-0 OOM reports or cache-capacity pressure; partially filled cold-image requests retain their host image rows until
they finish or are cancelled.

## Host checks and rollout

- Focused server/core tests: 99 passed (66 server and 33 engine seam).
- Broader Linux suite: candidate 227/229 passed; baseline 224/226 passed. Both have the same two failures:
  an unaligned direct-I/O read expectation and a draft-survival expectation of 0.5 despite the existing +0.19
  confidence calibration (actual 0.5473576181430894). The final cleanup-only incremental rerun reported 221/223
  with the same failures; the changed DeepSeek module was rerun. No added failure.
- Server goldens: 312/319 exact, seven recorded differences, zero unexpected differences. Existing allocator-leak
  diagnostics from the golden harness remain; this is not a leak-free claim.
- Lean checker: 2,201 baseline findings, 2,197 candidate findings, with no added comment-width finding.
- The published patches apply without the separate SSE/golden PR, after `0027`, and recreate all eight changed
  source files and the new regression file exactly.
- 1,295 of the 1,303 pre-existing source files are byte-identical; no pre-existing source file was removed.
- All four ranks were updated together after draining. Acceptance tests passed before traffic reopened.
  A three-minute stability watch passed with no supervisor restart.

Structured-output requests already returned HTTP 400 before this update; that separate engine limitation remains.

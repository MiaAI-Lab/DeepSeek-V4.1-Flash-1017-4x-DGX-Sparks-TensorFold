# Current-recipe MTU combination and output validation (2026-10-26)

Both images include the current recipe's `0026-rdma-active-mtu.patch` from `220547b`;
the only source delta between them is the proposed idle patch. The compiled source was checked against
the current recipe plus the proposed patch. All four GB10 nodes used matching image IDs in each run.
Tests used the existing jumbo-frame switchless ring, unchanged weights/rank caches, context 1,048,576,
pool 2,097,152, 16 streams, DSpark and CUDA graphs. This does not test a switched MTU-1500 fabric.

| Image | ID |
| --- | --- |
| MTU control, without idle patch | `sha256:cd45f1004a567ea95dcfab0913904fce21facbd38fc213d92d11d04d37a15bd5` |
| MTU plus idle patch | `sha256:07afd700102e1a9ae6a97cedd4985a8589a88f9e14fc23a9792fd88628cdd475` |

## Build safety and results

The images were built serially under `/tmp/codex-docker-build.lock`, with no active GPU compute,
`MAX_JOBS=1`, `NVCC_THREADS=1`, Zig `-j1`, and only sm_121. The 32-GiB MemAvailable floor was monitored.
Control starting/minimum MemAvailable was 116.69/114.07 GiB; patched was 116.37/114.04 GiB.
Daemon build records were terminal and native compilers absent before releasing the lock.
All three futex gate tests passed in the patched image. The same image pair was transferred to every node.

## End-to-end and fixed-input checks

The unchanged `tools/validate.py` from recipe `220547b` was run inside each image (Python/PyAV/NumPy present):

```bash
python3 tools/validate.py http://127.0.0.1:11435 --model deepseek-v4.1-flash --save control-replies.json
# Restart all ranks with the MTU+idle image and copy the reference replies into it.
python3 tools/validate.py http://127.0.0.1:11435 --model deepseek-v4.1-flash --save patched-replies.json --compare control-replies.json
```

Both runs exited zero, with no skipped checks:

- Text, thinking, tool call and tool result, image and video passed.
- The 21,875-token long prompt returned the expected number.
- Concurrent versus sequential greedy replies: 8/8 equal in each image.
- Patched versus saved control greedy reply texts: 8/8 exactly equal.

The eight comparison requests use the same fixed prompt list, temperature zero, thinking off and a
160-token maximum. Equality concerns the response text, not request IDs, timing fields or every possible
model output. [Raw checks, prompts, reply texts and SHA256 values](mtu-output-validation-20261026.json)
record the bounded result. No throughput improvement or universal output-equivalence claim is made.

## Idle after the complete suite

After another 30 seconds settling, a 60-second cgroup CPU window measured:

| NCCL rank | CPU |
| --- | ---: |
| 0 | 0.302% |
| 1 | 0.245% |
| 2 | 0.203% |
| 3 | 0.230% |

Inference counters were unchanged and health remained idle throughout. Runtime arguments and environment
values matched the earlier A/B experiment. The MTU+idle image remains running on all four nodes.
Wall-power attribution, full-window needle retrieval, packed Engram and switched-fabric validation remain
outside these tests. The PR remains Draft for feedback; this record does not assert those untested cases.

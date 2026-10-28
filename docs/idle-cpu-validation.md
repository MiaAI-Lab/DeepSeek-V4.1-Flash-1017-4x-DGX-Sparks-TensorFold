# Controlled idle CPU recheck (2026-10-26)

Both variants already use TensorFold recipe `6beaad9` and engine `5940208`.
A is without the idle patch; B adds only the proposed `0027` idle patch.
The image source manifests differ only in `idle_gate.zig`, `lanes.zig`, `native.zig` and `rdma.zig`.
Image environments, runtime environment names/values and launch arguments match.
All four nodes in each phase use the expected identical image ID. Model weights and rank caches are unchanged.
This is not a comparison of SGLang versus TensorFold or of recipe versions.

The order is A1 -> B1 -> A2 -> B2. Every phase starts all four containers afresh, completes
an identical `19 + 23` chat smoke request, waits 30 seconds, then samples container CPU for 60 seconds.
CPU comes from the cgroup-v2 `cpu.stat` cumulative `usage_usec` difference over monotonic elapsed time;
100% means one CPU core, not 100% of the 20-core host.

| NCCL rank | A1: unpatched | B1: patched | A2: unpatched | B2: patched |
| --- | ---: | ---: | ---: | ---: |
| 0 | 300.11% | 0.32% | 300.12% | 0.28% |
| 1 | 300.09% | 0.21% | 300.08% | 0.22% |
| 2 | 300.12% | 0.21% | 300.09% | 0.23% |
| 3 | 300.09% | 0.21% | 300.07% | 0.24% |

Every measurement window had unchanged prompt/generated-token counters, request-latency count and
client-disconnection count. Periodic `/health` checks showed no live/waiting requests or warmup.
HTTP logs contained zero POST requests during the windows. Container ID, initial PID, start time,
running state and restart count were checked before and after; no restart/recreation was observed.
The three busy threads in each A phase each consumed about one CPU core.

The original runner's final environment check compared array order rather than environment values
and falsely reported failure after all four windows completed. Independent post-validation verified
unique keys, identical names/values, identical arguments and all measurement validity checks above.
The CPU windows were retained unchanged; this was a validation-script correction, not a rerun or a
selection of successful samples. The patched server was restored afterwards.

[Machine-readable measurements](idle-cpu-ab-20261026.json) include elapsed times, raw CPU counters,
image IDs and inference counters. No TPS gain, output equivalence or isolated watts-saved figure
is claimed by this idle test. Live tests of the current recipe's MTU patch combination and the broader
feature/output validation remain pending at draft submission time.

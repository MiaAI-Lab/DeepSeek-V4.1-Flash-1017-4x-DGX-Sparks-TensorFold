# Engram's tables on ZFS — October 29, 2026

Measured on one DGX Spark (GB10) of this cluster, with the recipe's default Engram path (`ENGRAM_PACKED=0`: a row's
256-byte FP8 values and its 8 UE8M0 scales read as two O_DIRECT preads) and with packing on (`0003`: one 264-byte row,
taken through the 4 KiB-aligned span `core/direct_io.zig` requires).
Engine: BertholomusAI's `59402086381a42c74d53053443224eec2790b4c5` with the recipe patches. 64 readers in flight, GPU
idle, 256-byte reads at random record offsets.

The tables are about 203 GB of rows read at random, so they are a random-read workload on whatever holds them, and on
ZFS the dataset's own settings decide what each row costs. A pool's defaults are the wrong ones by a wide margin:
`recordsize=128K` and `compression=zstd` made every row read pull a whole compressed record, at about 500 rows a second
and 32 MB/s, and a 16k-token prompt took over 500 seconds to prefill.

| Dataset | Rows/s | Latency | CPU per row |
| --- | --- | --- | --- |
| `recordsize=128K compression=zstd` (defaults), 2 GiB ARC | ~500 | ~32 MB/s | not measured |
| `recordsize=16K encryption=aes-256-gcm` | 54,976 | 1,183 µs | 337 µs |
| `recordsize=16K encryption=off` | 189,815 | 337 µs | 46 µs |
| `recordsize=4K encryption=off` | 292,617 | 196 µs | 38 µs |

## The dataset

Keep the tables on their own dataset: the checkpoint is read sequentially once at load and is happy at a pool's
defaults, and changing `recordsize` on a dataset only affects blocks written after the change.

```sh
zfs create -o atime=off -o compression=off -o encryption=off -o primarycache=metadata \
  -o recordsize=4K -o secondarycache=none -o com.sun:auto-snapshot=false \
  -o mountpoint=/var/lib/dsv41-engram rpool/dsv41-engram
```

Point `ENGRAM_PACKED_DIR` (packed shards) or `WORKER_ENGRAM_DIR` (the tables) at that mount, and copy the tables there
before the first start; `scripts/prepare.sh` writes the packed shards where it is told, so a re-prepare after a
checkpoint change is the only cost.

| Setting | Why |
| --- | --- |
| `encryption=off` | ZFS native encryption on aarch64 runs the generic C GCM path; the accelerated one is x86-64 only (openzfs #9749, #10347, with #16601 and #19194 unmerged). It measured 7.5x the CPU and 3.5x the latency per block read, about 16 of the Spark's 20 cores at 50k rows a second, which starved the engine whenever a second request prefilled. An unencrypted child of an encrypted parent is legal (openzfs #8870), and the tables are model weights, not secrets. |
| `recordsize=4K` | A row is at most one 4 KiB block either way (the direct read's aligned span), so a larger record is pure amplification: 4K moved 4 KiB per row where 16K moved 16 KiB, for 1.5x the rows. The cost is indirect-block metadata, about 6.5 GiB at 4K against the 2 GiB ARC this was measured with; raise the ARC and 4K becomes unambiguously free. |
| `compression=off` | Dense FP8 does not compress: zle measured 1.00x over these tables, three to seven zero bytes a MiB and a longest run of six. Compression can also push a read off the direct-I/O path, and it costs about 1.21x the space. |
| `primarycache=metadata` | Rows are near-uniformly random, so caching data only churns a small ARC. The in-memory row cache `ENGRAM_CACHE_MIB` is the cache for these reads, not ZFS's. |
| `secondarycache=none` | The same argument for an L2ARC; do not ingest this dataset into one. |
| `atime=off` | A read must not become a write. |
| `com.sun:auto-snapshot=false` | A reproducible copy of a checkpoint's tables is never worth a snapshot. |

## What this does not fix

ZFS is not the only layer. On this kernel Linux AIO completes inside `io_submit` (1.1k reads a second at queue depth
1, re-measured 2026-10-26), so `ENGRAM_AIO=1`'s ring carries the concurrency and the reader pool's depth does the rest;
neither is a dataset setting.

A dataset's `recordsize` also cannot help a read that never reaches the disk, so check the descriptors before blaming
the pool: `tf-dsv41-engram-check ENGRAM_DIR --rows 4096` reports, per layer, whether the reader opened the tables
through O_DIRECT (`gather_rows2_direct_at`) or fell back to the buffered path, with each path's timing.
A filesystem that refuses O_DIRECT silently gives the buffered path, and the buffered path reads through the page
cache at the dataset's recordsize, which is where the defaults cost their seconds.

Provenance: one machine, one pool layout, the recipe's default Engram path and its packed alternative. Treat the
absolute row counts as one Spark's and the ratios as the finding.

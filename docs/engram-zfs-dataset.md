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
| `recordsize=128K compression=zstd` (defaults), 2 GiB ARC | ~500 (~32 MB/s) | not measured | not measured |
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

Point `ENGRAM_PACKED_DIR` (packed shards) or `WORKER_ENGRAM_DIR` (the workers' tables) at that mount;
`scripts/prepare.sh` writes to whichever you set, so a re-prepare after a checkpoint change is the only cost. Without
packing, rank 0 reads the tables from its own checkpoint folder (`HF_CACHE`), not from `WORKER_ENGRAM_DIR`.

| Setting | Why |
| --- | --- |
| `encryption=off` | ZFS native encryption on aarch64 runs the generic C GCM path; the accelerated one is x86-64 only (openzfs #9749, #10347; #16601 is still open). #19194, a Linux kernel crypto API backend, merged on 2026-10-23, after these numbers were taken: on a release that has it, measure again before giving up encryption. It measured 7.5x the CPU and 3.5x the latency per block read, about 16 of the Spark's 20 cores at 50k rows a second, which starved the engine whenever a second request prefilled. An unencrypted child of an encrypted parent is legal (openzfs #8870), and the tables are model weights, not secrets. |
| `recordsize=4K` | A row is at most one 4 KiB block either way (the direct read's aligned span), so a larger record is pure amplification: 4K moved 4 KiB per row where 16K moved 16 KiB, for 1.5x the rows. The cost is indirect-block metadata, about 6.5 GiB at 4K against the 2 GiB ARC this was measured with; raise the ARC and 4K becomes unambiguously free. |
| `compression=off` | Dense FP8 does not compress: zle measured 1.00x over these tables, three to seven zero bytes a MiB and a longest run of six. Compression can also push a read off the direct-I/O path, and it costs about 1.21x the space. |
| `primarycache=metadata` | Rows are near-uniformly random, so caching data only churns a small ARC. With `ENGRAM_PACKED=1`, the in-memory row cache `ENGRAM_CACHE_MIB` is the cache for these reads, not ZFS's. |
| `secondarycache=none` | The same argument for an L2ARC; do not ingest this dataset into one. |
| `atime=off` | A read must not become a write. |
| `com.sun:auto-snapshot=false` | A reproducible copy of a checkpoint's tables is never worth a snapshot. |

## Not a dataset property: the vdev's sync read limit

A dataset setting cannot help if the reads never reach the disk while they wait their turn, and on the recipe's path
they can. The tables are opened `O_DIRECT`, which ZFS issues as a **sync** read, and the per-vdev cap on sync reads is
`10` by default (`zfs_vdev_sync_read_max_active`). The reader pool is deeper than that for a reason -- its threads only
overlap if the vdev accepts them -- so at the default the pool queues behind the cap:

| `zfs_vdev_sync_read_max_active` / `_min_active` | 2048 random rows, layer 1 | rows/s |
| --- | --- | --- |
| 10 / 10 (default) | 27.2 ms | ~75,000 |
| **64 / 32** | **14.3 ms** | **~143,000** |
| 128 / 64 | 14.3 ms | ~143,000 |

Set it in the module options (e.g. `options zfs zfs_vdev_sync_read_max_active=64 zfs_vdev_sync_read_min_active=32`);
`zfs_vdev_sync_read_min_active` tracks it so the minimum does not starve the queue. Deeper than 64/32 bought nothing on
one NVMe. Two other knobs were tried against the same workload and did **not** move it: `direct=always` on the dataset
(ZFS 2.4 Direct IO, 14.0 ms) and `zfs_prefetch_disable=1` (14.4 ms; leave prefetch on, the checkpoint load uses it).

## What this does not fix

ZFS is not the only layer, and the one thing a dataset cannot give back is asynchronous kernel AIO. On this kernel
Linux AIO completes inside `io_submit` (1.1k reads a second at queue depth 1, re-measured 2026-10-26) because ZFS has
no `aio_read`/`aio_write` since the Linux 4.1 `read_iter` conversion, so `ENGRAM_AIO=1` as shipped runs a batch at depth
1: measured here, 2048 random rows took 2.4-3.9 s against 14 ms for the reader pool.

io_uring is the way out (patch `0055`): it hands a request the file system cannot complete in the caller's context to
the kernel's io-wq workers, so the batch overlaps whatever ZFS does. On this dataset you do not have to ask for it.
`TF_DS_ENGRAM_URING` unset means auto — the ring's first batch submits two reads and checks `io_getevents`: completions
already there mean `io_submit` never queued, which is exactly what ZFS does, and the ring switches to io_uring and says
so in the log; completions still in flight mean the AIO path is fine and it stays there. `=1` and `=0` force a backend.
Both paths are unit-tested row-for-row against the reader pool.

A dataset's `recordsize` also cannot help a read that never reaches the disk, so check the descriptors before blaming
the pool: `tf-dsv41-engram-check ENGRAM_DIR --rows 4096` reports, per layer, whether the reader opened the tables
through O_DIRECT (`gather_rows2_direct_at`) or fell back to the buffered path, with each path's timing.
A filesystem that refuses O_DIRECT silently gives the buffered path, and the buffered path reads through the page
cache at the dataset's recordsize, which is where the defaults cost their seconds.

Provenance: one machine, one pool layout, the recipe's default Engram path and its packed alternative; the vdev and
`direct` numbers above come from one Spark with the tables on their dataset (median of six runs each, 2026-11-06).
Treat the absolute row counts as one Spark's and the ratios as the finding.

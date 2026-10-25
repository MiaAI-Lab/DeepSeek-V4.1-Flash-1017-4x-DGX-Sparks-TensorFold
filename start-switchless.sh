#!/usr/bin/env bash
# Serve DeepSeek-V4.1-Flash-1017 on four DGX Sparks cabled as a ring without a switch: ./start.sh for that fabric. Cable
# spark1 - spark2 - spark3 - spark4 - spark1 (each Spark's two CX7 ports to its two neighbours, one subnet per cable)
# and list WORKER .. WORKER3 in scripts/local.sh in the ring's order. scripts/nodes.sh finds the ring and writes the
# NCCL graph files of every rank and its two-rank links; ORDER=auto places the 2D split's nodes 0, 1, 3, 2 around it
# (every row and column partner on a cable), and the Spark across the ring is reached through a neighbour. Four Sparks
# on a switch: ./start.sh; this script stops there and says so.
#
# Usage: ./start-switchless.sh [restart] [extra tensorfold-native serve args]   (as ./start.sh; ./stop.sh stops it)
#   DRY_RUN=1 ./start-switchless.sh   # print the ring found and every rank's docker command, change nothing
set -euo pipefail
case "${1:-}" in
  help|-h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; echo; echo "./start.sh's options:"; echo ;;
esac
export FABRIC_EXPECT=ring
exec "$(dirname "$(readlink -f "$0")")/start.sh" "$@"

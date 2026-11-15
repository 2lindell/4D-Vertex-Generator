#!/bin/sh
# The crossing search run alongside the ixoic closure (2 workers): decaic, contic, then hyic ix1-9. Sheets done here are
# skipped by run_queue2.sh's later pass (crossing_search.py skips any sheet whose output exists).
cd "$(dirname "$0")"
echo "start (parallel) $(date -u)" >> crossing_search.log
NPROC=2 nice python crossing_search.py decaic ki io >> crossing_search.log 2>&1
NPROC=2 nice python crossing_search.py contic ox oxb tox icp icpb ki io >> crossing_search.log 2>&1
NPROC=2 nice python crossing_search.py hyic ix0 ix1 ix2 ix3 ix4 ix5 ix6 ix7 ix8 ix9 >> crossing_search.log 2>&1
python crossing_search.py summary > /dev/null 2>&1
echo "PARALLEL CROSSING SEARCH DONE $(date -u)" >> crossing_search.log

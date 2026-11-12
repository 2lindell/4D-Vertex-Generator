#!/bin/sh
# Queued: starts when one of the running jobs (the centre recompute, or the ixoic pipeline) has finished; resumable
# (sheets already done are skipped), so after a restart just run it again.
cd "$(dirname "$0")"
while pgrep -f "recentre.py" > /dev/null && pgrep -f "run_rest.sh" > /dev/null; do sleep 300; done
echo "start $(date -u)" >> crossing_search.log
NPROC=3 nice python crossing_search.py contic ox oxb tox icp icpb ki io >> crossing_search.log 2>&1
NPROC=3 nice python crossing_search.py decaic ki io >> crossing_search.log 2>&1
NPROC=3 nice python crossing_search.py hyic ix0 ix1 ix2 ix3 ix4 ix5 ix6 ix7 ix8 ix9 >> crossing_search.log 2>&1
python crossing_search.py summary > /dev/null 2>&1
echo "CROSSING SEARCH DONE $(date -u)" >> crossing_search.log

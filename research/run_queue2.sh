#!/bin/sh
# Queued after the ixoic wall refit (ixoic_walls.py) and the ixoic pipeline (run_rest.sh) have both finished:
# the crossing search (contic, decaic, ixitic), then the missing ixoic walls (hyic_atlas/missing_walls.py), then the
# crossing search on the ixoic sheets (with the walls found). Resumable: every stage skips the sheets it has done.
cd "$(dirname "$0")"
while pgrep -f "ixoic_walls.py" > /dev/null || pgrep -f "run_rest.sh" > /dev/null; do sleep 300; done
echo "start $(date -u)" >> crossing_search.log
NPROC=3 nice python crossing_search.py contic ox oxb tox icp icpb ki io >> crossing_search.log 2>&1
NPROC=3 nice python crossing_search.py decaic ki io >> crossing_search.log 2>&1
NPROC=3 nice python crossing_search.py hyic ix0 ix1 ix2 ix3 ix4 ix5 ix6 ix7 ix8 ix9 >> crossing_search.log 2>&1
python crossing_search.py summary > /dev/null 2>&1
echo "CROSSING SEARCH DONE $(date -u)" >> crossing_search.log
(cd hyic_atlas && NPROC=3 nice python missing_walls.py >> missing_walls.log 2>&1)
echo "MISSING WALLS DONE $(date -u)" >> hyic_atlas/missing_walls.log
NPROC=3 nice python crossing_search.py hyic ixo0 ixo1 ixo2 ixo3 ixo4 ixo5 ixo6 ixo7 ixo8 ixo9 >> crossing_search.log 2>&1
python crossing_search.py summary > /dev/null 2>&1
echo "IXOIC CROSSING SEARCH DONE $(date -u)" >> crossing_search.log

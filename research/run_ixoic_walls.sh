#!/bin/sh
# Queued: the ixoic wall equations once the last sheet's map exists, then the crossing search on the ixoic sheets once
# the main crossing search is done. Resumable: just run it again after a restart.
cd "$(dirname "$0")"
until [ -e hyic_atlas/snub_map_ixo9.json ]; do sleep 300; done
(cd hyic_atlas && NPROC=2 nice python ixoic_walls.py >> ixoic_walls.log 2>&1)
echo "IXOIC WALLS DONE" >> hyic_atlas/ixoic_walls.log
until grep -q "CROSSING SEARCH DONE" crossing_search.log 2>/dev/null; do sleep 300; done
NPROC=3 nice python crossing_search.py hyic ixo0 ixo1 ixo2 ixo3 ixo4 ixo5 ixo6 ixo7 ixo8 ixo9 >> crossing_search.log 2>&1
python crossing_search.py summary > /dev/null 2>&1
echo "IXOIC CROSSING SEARCH DONE $(date -u)" >> crossing_search.log

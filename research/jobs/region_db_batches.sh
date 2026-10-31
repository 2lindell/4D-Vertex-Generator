#!/bin/bash
# The region database for F2's neighbouring walls, three walls at a time; a finished batch is recorded in
# state/region_db.done and skipped next time, so a restart costs at most the batch in progress (whose classifier
# labels are cached as they go).
cd "$(dirname "$0")/../swirlprism_atlas"
J="$(cd ../jobs && pwd)"
for batch in F1,F4,F5 X15,X17,X19 X20,X22,X23 X25,X27,X28 X33,X34,X39 X47,X48,X52 X7,X82,X83; do
  grep -qx "$batch" "$J/state/region_db.done" 2>/dev/null && continue
  echo "== $batch $(date +%T)" >> "$J/logs/region_db.log"
  python3 region_db.py "$batch" >> "$J/logs/region_db.log" 2>&1 && echo "$batch" >> "$J/state/region_db.done"
done
grep -qx "all" "$J/state/region_db.done" || { python3 region_view.py --export >> "$J/logs/region_db.log" 2>&1 && echo all >> "$J/state/region_db.done"; }

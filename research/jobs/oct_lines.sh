#!/bin/bash
# The octahedral swirl's lines and points where walls meet (atlas_model.py, which resumes from its partial file),
# then the page's data (make_atlas_oct.py) and the page (build_page.py).
cd "$(dirname "$0")/../tetraswirl_atlas"
J="$(cd ../jobs && pwd)"
SWIRL_GROUP=oct python3 -u atlas_model.py >> "$J/logs/oct_lines.log" 2>&1 \
  && SWIRL_GROUP=oct python3 make_atlas_oct.py >> "$J/logs/oct_lines.log" 2>&1 \
  && SWIRL_GROUP=oct python3 build_page.py >> "$J/logs/oct_lines.log" 2>&1 \
  && echo done > "$J/state/oct_lines.done"

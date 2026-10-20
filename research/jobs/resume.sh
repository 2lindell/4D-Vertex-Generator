#!/bin/bash
# Start every unfinished long job that is not already running (run at session start, and by the hourly check-in).
# Each job skips the work it has finished, so running this again is always safe.
J="$(cd "$(dirname "$0")" && pwd)"
start() {   # name, script, marker of a finished job, process pattern
  [ -f "$J/state/$3" ] && grep -qx "${5:-done}" "$J/state/$3" && { echo "$1: finished"; return; }
  if pgrep -f "$4" > /dev/null; then echo "$1: running"; return; fi
  setsid nohup bash "$J/$2" > /dev/null 2>&1 < /dev/null &
  echo "$1: started"
}
start "region database" region_db_batches.sh region_db.done "region_db.py|region_db_batches.sh" all
start "octahedral lines and points" oct_lines.sh oct_lines.done "atlas_model.py|oct_lines.sh"

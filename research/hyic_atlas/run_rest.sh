#!/bin/sh
# Resumable: every stage is skipped when its output exists (strata.py also checkpoints each stratum), so after a
# container restart just run this again.
cd "$(dirname "$0")"
step() {  # step <output file> <sub> <command...>
  out=$1; sub=$2; shift 2
  [ -e "$out" ] && return 0
  HYIC_SUB=$sub "$@"
}
(
  for s in ix0 ix1 ix2 ix3 ix4 ix5 ix6 ix7 ix8 ix9; do
    step surface_survey_$s.json $s python surface_survey.py 300 >> surface_survey.log 2>&1; done
  for s in ix0 ix1 ix2 ix3 ix4 ix5 ix6 ix7 ix8 ix9; do step closure_$s.json $s python closure.py >> closure.log 2>&1; done
  echo "SEARCH DONE" >> surface_survey.log
) &
(
  for s in ixo0 ixo1 ixo2 ixo3 ixo4 ixo5 ixo6 ixo7 ixo8 ixo9; do
    step snub_map_$s.json $s python snub_map.py 18 > snub_map_$s.log 2>&1
    step wall_shapes_$s.json $s python wall_shapes.py > wall_shapes_$s.log 2>&1
    step strata_$s.json $s python strata.py 24 >> strata_$s.log 2>&1
    step survey_$s.json $s python survey.py 100 > survey_$s.log 2>&1
    echo "$s done" >> run_sheets.log
  done
  echo "ALL DONE" >> run_sheets.log
  [ -e ixoic_shapes.json ] || python ixitic_shapes.py ixo > ixoic_merge.log 2>&1
  for s in ixo0 ixo1 ixo2 ixo3 ixo4 ixo5 ixo6 ixo7 ixo8 ixo9; do step closure_$s.json $s python closure.py >> closure_ixoic.log 2>&1; done
  echo "IXOIC CLOSURE DONE" >> closure_ixoic.log
) &
wait

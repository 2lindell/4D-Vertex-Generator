#!/bin/sh
# (restart after the container restart: the rest of the ixitic surface survey and the ixitic closure check; and in
#  parallel the rest of ixoic sheet 0, sheets 1-9, their merge and the ixoic closure check)
cd "$(dirname "$0")"
(
  for s in ix5 ix6 ix7 ix8 ix9; do HYIC_SUB=$s python surface_survey.py 300 >> surface_survey.log 2>&1; done
  for s in ix0 ix1 ix2 ix3 ix4 ix5 ix6 ix7 ix8 ix9; do HYIC_SUB=$s python closure.py >> closure.log 2>&1; done
  echo "SEARCH DONE" >> surface_survey.log
) &
(
  HYIC_SUB=ixo0 python strata.py 24 > strata_ixo0.log 2>&1
  HYIC_SUB=ixo0 python survey.py 100 > survey_ixo0.log 2>&1
  echo "ixo0 done" >> run_sheets.log
  for s in ixo1 ixo2 ixo3 ixo4 ixo5 ixo6 ixo7 ixo8 ixo9; do
    HYIC_SUB=$s python snub_map.py 18 > snub_map_$s.log 2>&1
    HYIC_SUB=$s python wall_shapes.py > wall_shapes_$s.log 2>&1
    HYIC_SUB=$s python strata.py 24 > strata_$s.log 2>&1
    HYIC_SUB=$s python survey.py 100 > survey_$s.log 2>&1
    echo "$s done" >> run_sheets.log
  done
  echo "ALL DONE" >> run_sheets.log
  python ixitic_shapes.py ixo > ixoic_merge.log 2>&1
  for s in ixo0 ixo1 ixo2 ixo3 ixo4 ixo5 ixo6 ixo7 ixo8 ixo9; do HYIC_SUB=$s python closure.py >> closure_ixoic.log 2>&1; done
  echo "IXOIC CLOSURE DONE" >> closure_ixoic.log
) &
wait

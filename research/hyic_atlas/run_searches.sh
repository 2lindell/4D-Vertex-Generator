#!/bin/sh
# (after the ixoic sheets: the targeted surface survey, then the closure check, on the ten ixitic sheets)
cd "$(dirname "$0")"
until grep -q "ALL DONE" run_sheets.log; do sleep 60; done
for s in ix0 ix1 ix2 ix3 ix4 ix5 ix6 ix7 ix8 ix9; do HYIC_SUB=$s python surface_survey.py 300 >> surface_survey.log 2>&1; done
for s in ix0 ix1 ix2 ix3 ix4 ix5 ix6 ix7 ix8 ix9; do HYIC_SUB=$s python closure.py >> closure.log 2>&1; done
echo "SEARCH DONE" >> surface_survey.log

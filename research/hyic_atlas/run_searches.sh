#!/bin/sh
# (after the optimized re-run (repolish.py) and the central re-run (recentral.py): the targeted surface survey, then
#  the closure check, on the ten ixitic sheets -- alongside the ixoic sheets)
cd "$(dirname "$0")/.."
until grep -q "REPOLISH DONE" repolish.log; do sleep 60; done
python recentral.py > recentral.log 2>&1
python toxitic_atlas/optimize_central.py gather >> recentral.log 2>&1
python hyic_atlas/ixitic_offs.py gather >> recentral.log 2>&1
python build_off_readmes.py >> recentral.log 2>&1
echo "RECENTRAL DONE" >> recentral.log
cd hyic_atlas
for s in ix0 ix1 ix2 ix3 ix4 ix5 ix6 ix7 ix8 ix9; do HYIC_SUB=$s python surface_survey.py 300 >> surface_survey.log 2>&1; done
for s in ix0 ix1 ix2 ix3 ix4 ix5 ix6 ix7 ix8 ix9; do HYIC_SUB=$s python closure.py >> closure.log 2>&1; done
echo "SEARCH DONE" >> surface_survey.log

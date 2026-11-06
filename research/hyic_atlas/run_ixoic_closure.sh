#!/bin/sh
# (after the ten ixoic sheets: merge them (ixitic_shapes.py ixo), then the closure check on each ixoic sheet)
cd "$(dirname "$0")"
until grep -q "ALL DONE" run_sheets.log; do sleep 60; done
python ixitic_shapes.py ixo > ixoic_merge.log 2>&1
for s in ixo0 ixo1 ixo2 ixo3 ixo4 ixo5 ixo6 ixo7 ixo8 ixo9; do HYIC_SUB=$s python closure.py >> closure_ixoic.log 2>&1; done
echo "IXOIC CLOSURE DONE" >> closure_ixoic.log

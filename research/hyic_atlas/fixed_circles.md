# Ixoic half-turn lines traced whole

fixed_lines.py sampled each line piece by piece, one sheet's chamber at a time (120 points each). Here each line is
taken whole: a half-turn g fixes a plane of seed vectors x (g x = x), a circle of seeds; the central inversion is in
ixoic, so x and -x give the same polychoron and half the circle is the whole line. The half-turns fall into two
conjugacy classes (180 and 90 of them), so two circles cover every half-turn line of every sheet. Each circle is
sampled at N angles (labelled at hull tolerance 1e-7, the seed beta = X^-1 x taken with whichever sign sums positive),
every change of key is bisected to 1e-12 in angle, and a key met only at a bisection, different from both sides, is
a point. The keys are then compared with what the atlas recorded (fixed_lines.json, ixoic_shapes.json, the closure
checks): any key not there was missed.

## Half-turns of the class of 90

2000 samples round the line (0 failed to label), 24 changes of key, 25 stretches, 4 points between stretches. Keys met: 7; recorded by fixed_lines.py or the atlas: 6; not recorded: 1 (f86ef4998f60).

## Half-turns of the class of 180

2000 samples round the line (1 failed to label), 21 changes of key, 22 stretches, 5 points between stretches. Keys met: 9; recorded by fixed_lines.py or the atlas: 9; not recorded: 0.


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


## Outcome

Every change of key round both lines was then bisected and the polychoron there classified, loosening the hull
tolerance from 1e-9 to 1e-4 until it gave a third key. Five of the six points between stretches were already recorded
(2f41ae404d62, 524c547d9ed6, 1e8454f185ca Regposbitad, 8ef1c968ad64 Petsobted, 992e194bf999); two changes have no
point between (01f527 to 85a06f, 612fa0 to 970053). One was missed: **f86ef4998f60**, between 7465267cd9b6 and
9fc73ba17c51 on the half-turn line of the class of 90, exactly at beta = (0, sqrt2 - 1, 2 - sqrt2, 0) on the chamber
edge b1 = b4 = 0 (sheets ixo7, ixo9; (0, 0.75787, 0, 0.24213) in ixo6, ixo8). There it classifies cleanly at every
tolerance from 1e-10 to 1e-7; the strata and fixed_lines.py bisected the change only to points near it, where the hull
could not be classified, so neither recorded it. It is a point (point_check.py) and the list's **Bamtuted**
(1440 vertices, edges 455,3^6: 240 gyrotuts, 480 trigyps), which ixoic_match.py had as not found. It is added to
fixed_lines.json (sheet ixo7). The key f86ef4 the scan first met, 3e-7 from the change at tolerance 1e-7, was this same
polychoron seen through the tolerance.

# Optimized seeds and .off files

The most even seed of each shape: the one that minimises R = longest edge / shortest edge (decaic_atlas/optimize_seeds.py, as swirlprism_atlas/exact_seeds.py; contic_atlas/optimize_seeds.py runs it on the contic groups). Edge lengths squared are quadratic forms in beta; the minimum is found on the shape's locus (its mirror / wall equations) while every sign condition of the shape holds, then the active equations are solved (40 digits for the decaic family; double precision for the contic family, whose forms carry sqrt2).

* **minimum inside the shape**: attained; R rises at every point of a ring of small steps round it on the locus ('not unique': R is flat along some direction there, a curve of equally even seeds).
* **infimum on the shape's edge**: R falls all the way to the shape's boundary (often a mirror, where a snub becomes the Wythoffian shape): never attained by the shape itself. Its .off uses a seed just inside (column 'off seed', with its R).
* **no progress**: no feasible step from the seed was found; the seed is kept.
* Snub-group shapes on the mirrors are the full group's shapes (same orbits): their files are in the full group's folder.

A shape that lies on more than one face of the domain (sabipard: on b2 = 0 and on b3 = 0) is optimized on every face, the best kept (sabipard: R = (4 + sqrt30)/7 = 1.3538894 on b3 = 0; 1.5811388 = sqrt(5/2) on b2 = 0).

Folders (tessic family: seed vectors in R^4, the B4 frame, vertices the signed permutations of the group): decaic, kidecaic, iodecaic (seed vectors in R^5: vertices sign * x[perm]); contic, kicontic, iocontic (seed vectors in R^4, the standard F4 frame). Kicontic / iocontic shape names are by map order (no accepted list); their decaic counterparts are in contic_atlas/contic_snub_shapes.md.

## Decaic

| shape | kind | R start | R best | status | best beta | off seed (R) | file |
|---|---|---|---|---|---|---|---|
| sabotid | region (3) | 37.652747 | 1.8451542547 | minimum inside the shape | [0.324262604, 0.324262604, 0.175737396, 0.175737396] | same | sabotid_240v_140c_opt.off |
| gabotid | region (3) | 3.897613 | 2.2909944487 | minimum inside the shape | [0.348070239, 0.151929761, 0.348070239, 0.151929761] | same | gabotid_240v_120c_opt.off |
| gabipard | wall (2) | 19.000000 | 3.0000000000 | infimum on the shape's edge (a sign condition is active): not attained | [0.6, 0.0, 0.2, 0.2] | [0.6001916674, 0.0, 0.1998416661, 0.1999666665] (3.003336) | gabipard_120v_140c_opt.off |
| mobcated | wall (2) | 3.000000 | 1.7071067812 | infimum on the shape's edge (a sign condition is active): not attained | [0.3431457505, 0.2426406871, 0.4142135624, 0.0] | [0.3431359381, 0.2425647129, 0.4142993491, 0.0] (1.707995) | mobcated_120v_100c_opt.off |
| sobcated | wall (2) | 1.732051 | 1.2110322250 | minimum inside the shape | [0.3538893679, 0.3538893679, 0.2922212643, 0.0] | same | sobcated_120v_140c_opt.off |
| mabipard | wall (2) | 3.000000 | 1.7207592201 | minimum inside the shape | [0.3874258867, 0.0, 0.3874258867, 0.2251482266] | same | mabipard_120v_120c_opt.off |
| sabipard | wall (2) | 3.000000 | 1.3538893679 | minimum inside the shape | [0.3651483717, 0.3651483717, 0.0, 0.2697032567] | same | sabipard_120v_140c_opt.off |
| gobcated | wall (2) | 9.000000 | 4.0000000000 | infimum on the shape's edge (a sign condition is active): not attained | [0.1666666667, 0.1666666667, 0.6666666667, 0.0] | [0.166583333, 0.1666666667, 0.6667500003, 0.0] (4.002501) | gobcated_120v_120c_opt.off |
| petbotid | wall (2) | 1.767767 | 1.7071067812 | minimum inside the shape | [0.3963146941, 0.1603926828, 0.2738074364, 0.1694851868] | same | petbotid_240v_50c_opt.off |
| tetbotid | wall (2) | 1.732051 | 1.5773502692 | minimum inside the shape | [0.2376286257, 0.1753932606, 0.4363276016, 0.1506505121] | same | tetbotid_240v_60c_opt.off |
| bited | line (1) | 5.000000 | 1.1832159566 | minimum inside the shape | [0.5, 0.5, 0.0, 0.0] | same | bited_40v_120c_opt.off |
| biped | line (1) | 13.000000 | 2.2909944487 | minimum inside the shape | [0.696140478, 0.0, 0.0, 0.303859522] | same | biped_40v_60c_opt.off |
| bimted | line (1) | 3.000000 | 2.5811388301 | minimum inside the shape | [-0.0, 0.2792407799, 0.7207592201, 0.0] | same | bimted_60v_30c_opt.off |
| gabred | line (1) | 7.000000 | 3.0000000000 | infimum on the shape's edge (a sign condition is active): not attained | [0.25, 0.0, 0.75, 0.0] | [0.2498749995, 0.0, 0.7501250005, 0.0] (3.002001) | gabred_60v_120c_opt.off |
| mabred | line (1) | 3.363636 | 2.0000000000 | infimum on the shape's edge (a sign condition is active): not attained | [0.6666666667, 0.0, 0.3333333333, 0.0] | [0.6667708345, 0.0, 0.3332291655, 0.0] (2.000938) | mabred_60v_140c_opt.off |
| sabred | line (1) | 1.426935 | 1.2909944487 | minimum inside the shape | [0.5, 0.0, 0.5, 0.0] | same | sabred_60v_100c_opt.off |
| tabipard | line (1) | 7.505553 | 1.5773502692 | minimum inside the shape | [0.3660254038, 0.0, 0.4019237886, 0.2320508076] | same | tabipard_120v_60c_opt.off |
| gippid | line (1) | 11.000000 | 1.0000000000 | minimum inside the shape | [0.25, 0.25, 0.25, 0.25] | same | gippid_120v_30c_opt.off |
| tadeca | line (1) | 2.000000 | 1.7320508076 | minimum inside the shape (not unique: R is flat along the locus there) | [0.1342942076, 0.2314115848, 0.6342942076, 0.0] | same | tadeca_120v_40c_opt.off |
| tabcated | line (1) | 8.838835 | 1.7071067812 | minimum inside the shape | [0.3431457505, 0.2426406871, 0.4142135624, 0.0] | same | tabcated_120v_50c_opt.off |
| tispid | line (1) | 1.500000 | 1.4142135624 | minimum inside the shape (not unique: R is flat along the locus there) | [0.5857869049, 0.0, 0.1715738097, 0.2426392854] | same | tispid_120v_50c_opt.off |
| bideca | point (0) | 1.290994 | 1.2909944487 | point (no freedom) | [1.0, 0.0, 0.0, 0.0] | same | bideca_10v_30c_opt.off |
| spid | point (0) | 1.000000 | 1.0000000000 | point (no freedom) | [0.5, 0.0, 0.0, 0.5] | same | spid_20v_30c_opt.off |
| apid | point (0) | 1.581139 | 1.5811388301 | point (no freedom) | [0.0, 0.0, 1.0, 0.0] | same | apid_20v_30c_opt.off |
| deca | point (0) | 1.000000 | 1.0000000000 | point (no freedom) | [0.0, 0.5, 0.5, 0.0] | same | deca_30v_10c_opt.off |
| redeca | point (0) | 1.732051 | 1.7320508076 | point (no freedom) | [0.25, 0.0, 0.75, 0.0] | same | redeca_60v_40c_opt.off |
| respid | point (0) | 1.414214 | 1.4142135624 | point (no freedom) | [0.6666666667, 0.0, 0.3333333333, 0.0] | same | respid_60v_50c_opt.off |

## Kidecaic

| shape | kind | R start | R best | status | best beta | off seed (R) | file |
|---|---|---|---|---|---|---|---|
| mebosid | region (3) | 25.463868 | 1.8708286934 | infimum on the shape's edge (a sign condition is active): not attained | [0.2792407799, -0.0, 0.4415184401, 0.2792407799] | [0.2788728971, 4.05602e-05, 0.4422532627, 0.27883328] (1.875087) | mebosid_120v_560c_opt.off |
| new_region_01_120v_560c | region (3) | 15.840267 | 6.3256565248 | infimum on the shape's edge (a sign condition is active): not attained | [0.1245463961, 0.1245463961, 0.7296479065, 0.0212593014] | [0.0901207565, 0.1017394826, 0.7849696088, 0.0231701521] (9.032302) | new_region_01_120v_560c_120v_560c_opt.off |
| new_region_02_120v_680c | region (3) | 15.306752 | 6.3256565248 | infimum on the shape's edge (a sign condition is active): not attained | [0.1245463961, 0.1245463961, 0.7296479065, 0.0212593014] | [0.1244719184, 0.1245719694, 0.7296926768, 0.0212634354] (6.329775) | new_region_02_120v_680c_120v_680c_opt.off |
| new_region_03_120v_680c | region (3) | 14.469616 | 4.5825756950 | infimum on the shape's edge (a sign condition is active): not attained | [0.1666666667, 0.1666666667, 0.6666666667, 0.0] | [0.1135314351, 0.23878307, 0.6361131751, 0.0115723198] (6.863798) | new_region_03_120v_680c_120v_680c_opt.off |
| new_region_04_120v_560c | region (3) | 14.401088 | 2.7165684733 | infimum on the shape's edge (a sign condition is active): not attained | [0.2665823566, 0.1205963407, 0.4590602489, 0.1537610538] | [0.095644216, 0.3016208508, 0.5503223439, 0.0524125893] (6.860303) | new_region_04_120v_560c_120v_560c_opt.off |
| apbosid | region (3) | 41.702218 | 2.2671357138 | infimum on the shape's edge (a sign condition is active): not attained | [0.4394656318, 0.0, 0.3445462982, 0.21598807] | [0.4390762378, 0.0004497643, 0.3446765232, 0.2157974747] (2.269071) | apbosid_120v_420c_opt.off |
| sabosid | region (3) | 27.302256 | 1.6652318805 | minimum inside the shape | [0.3962831202, 0.3101654558, 0.1489627514, 0.1445886726] | same | sabosid_120v_440c_opt.off |
| gabosid | region (3) | 20.286319 | 1.8985107882 | infimum on the shape's edge (a sign condition is active): not attained | [0.4262441625, 0.2400881471, 0.2400881471, 0.0935795433] | [0.4258745808, 0.2403087959, 0.2403088349, 0.0935077884] (1.901003) | gabosid_120v_560c_opt.off |
| new_region_05_120v_560c | region (3) | 10.317586 | 2.5155111470 | infimum on the shape's edge (a sign condition is active): not attained | [0.5120065698, 0.0809138827, 0.2035397738, 0.2035397738] | [0.5119191832, 0.0809331484, 0.2034614118, 0.2036862566] (2.517293) | new_region_05_120v_560c_120v_560c_opt.off |
| new_region_06_120v_560c | region (3) | 15.663981 | 1.8985107882 | infimum on the shape's edge (a sign condition is active): not attained | [0.4262441625, 0.2400881471, 0.2400881471, 0.0935795433] | [0.5708700541, 0.0304568462, 0.0304930643, 0.3681800353] (12.869397) | new_region_06_120v_560c_120v_560c_opt.off |
| new_region_07_120v_680c | region (3) | 10.099666 | 3.1622776602 | infimum on the shape's edge (a sign condition is active): not attained | [0.6, -0.0, 0.2, 0.2] | [0.5751686008, 0.0245567534, 0.0544801304, 0.3457945154] (9.578563) | new_region_07_120v_680c_120v_680c_opt.off |
| new_region_08_120v_680c | region (3) | 10.401016 | 5.4848991540 | infimum on the shape's edge (a sign condition is active): not attained | [0.7238222469, 0.0222340399, 0.1216464783, 0.1322972348] | [0.6302684585, 0.0250873021, 0.0544924632, 0.2901517762] (9.846168) | new_region_08_120v_680c_120v_680c_opt.off |
| new_region_09_120v_680c | region (3) | 5.851483 | 3.1622950806 | infimum on the shape's edge (a sign condition is active): not attained | [0.6000012617, 6.882e-07, 0.1999988524, 0.1999991977] | [0.6000256547, 2.59567e-05, 0.1998990578, 0.2000493307] (3.163869) | new_region_09_120v_680c_120v_680c_opt.off |
| sigwebosid | wall (2) | 26.398538 | 1.8708286934 | infimum on the shape's edge (a sign condition is active): not attained | [0.2792407799, -0.0, 0.4415184401, 0.2792407799] | [0.2789856633, 2.53719e-05, 0.4420033016, 0.2789856633] (1.873520) | sigwebosid_120v_290c_opt.off |
| new_wall_01_120v_380c | wall (2) | 15.904112 | 6.3256565248 | infimum on the shape's edge (a sign condition is active): not attained | [0.1245463961, 0.1245463961, 0.7296479065, 0.0212593014] | [0.1244737874, 0.1244851692, 0.7297611927, 0.0212798506] (6.329660) | new_wall_01_120v_380c_120v_380c_opt.off |
| new_wall_02_120v_440c | wall (2) | 12.750220 | 6.3256565248 | infimum on the shape's edge (a sign condition is active): not attained | [0.1245463961, 0.1245463961, 0.7296479065, 0.0212593014] | [0.124484393, 0.1245450951, 0.7296988088, 0.0212717031] (6.329006) | new_wall_02_120v_440c_120v_440c_opt.off |
| new_wall_03_120v_500c | wall (2) | 11.362957 | 6.3256565248 | infimum on the shape's edge (a sign condition is active): not attained | [0.1245463961, 0.1245463961, 0.7296479065, 0.0212593014] | [0.1244863582, 0.124669552, 0.729571806, 0.0212722838] (6.328511) | new_wall_03_120v_500c_120v_500c_opt.off |
| new_wall_04_120v_500c | wall (2) | 11.503736 | 4.5825756950 | infimum on the shape's edge (a sign condition is active): not attained | [0.1666666667, 0.1666666667, 0.6666666667, 0.0] | [0.1665628116, 0.1668113246, 0.6666100888, 1.57749e-05] (4.585668) | new_wall_04_120v_500c_120v_500c_opt.off |
| tetbosid | wall (2) | 11.692719 | 1.9139484184 | minimum inside the shape | [0.2968602152, 0.0871919302, 0.4350354939, 0.1809123607] | same | tetbosid_120v_240c_opt.off |
| petbosid | wall (2) | 33.873448 | 1.9834008275 | minimum inside the shape | [0.4149782763, 0.1566487331, 0.2999027519, 0.1284702387] | same | petbosid_120v_230c_opt.off |
| gigwebosid | wall (2) | 20.272881 | 1.7207592201 | minimum inside the shape | [0.4007450022, 0.2833695086, 0.2833695086, 0.0325159806] | same | gigwebosid_120v_320c_opt.off |
| new_wall_05_120v_380c | wall (2) | 9.715348 | 3.4205262753 | infimum on the shape's edge (a sign condition is active): not attained | [0.1875, 0.1875, 0.5625, 0.0625] | [0.1873831933, 0.1875763659, 0.5625606766, 0.0624797642] (3.423122) | new_wall_05_120v_380c_120v_380c_opt.off |
| new_wall_06_120v_380c | wall (2) | 10.781103 | 2.5155111470 | minimum inside the shape | [0.5120065698, 0.0809138827, 0.2035397738, 0.2035397738] | same | new_wall_06_120v_380c_120v_380c_opt.off |
| new_wall_07_120v_380c | wall (2) | 6.370752 | 5.4848991540 | infimum on the shape's edge (a sign condition is active): not attained | [0.7238222469, 0.0222340399, 0.1216464783, 0.1322972348] | [0.7238213868, 0.0222383484, 0.1216250425, 0.1323152223] (5.485675) | new_wall_07_120v_380c_120v_380c_opt.off |
| new_wall_08_120v_440c | wall (2) | 10.058360 | 3.1622776602 | infimum on the shape's edge (a sign condition is active): not attained | [0.6, -0.0, 0.2, 0.2] | [0.5999727462, 1.80112e-05, 0.1998495238, 0.2001597187] (3.164640) | new_wall_08_120v_440c_120v_440c_opt.off |
| new_wall_09_120v_440c | wall (2) | 12.298535 | 5.4848991540 | infimum on the shape's edge (a sign condition is active): not attained | [0.7238222469, 0.0222340399, 0.1216464783, 0.1322972348] | [0.7236551003, 0.0222617295, 0.1215611942, 0.1325219759] (5.486726) | new_wall_09_120v_440c_120v_440c_opt.off |
| new_wall_10_120v_500c | wall (2) | 10.338378 | 5.4848991540 | infimum on the shape's edge (a sign condition is active): not attained | [0.7238222469, 0.0222340399, 0.1216464783, 0.1322972348] | [0.7237054183, 0.0222529486, 0.1215702819, 0.1324713512] (5.486881) | new_wall_10_120v_500c_120v_500c_opt.off |
| new_wall_11_120v_500c | wall (2) | 6.120185 | 3.1622776602 | infimum on the shape's edge (a sign condition is active): not attained | [0.6, -0.0, 0.2, 0.2] | [0.6000268786, 2.27109e-05, 0.199894808, 0.2000556026] (3.163978) | new_wall_11_120v_500c_120v_500c_opt.off |
| new_region_10_120v_680c | region (3) | 9.215822 | 4.5827142798 | infimum on the shape's edge (a sign condition is active): not attained | [0.1666620648, 0.1666620648, 0.66667178, 4.0905e-06] | [0.095969056, 0.2830745251, 0.6091285608, 0.0118278581] (8.166715) | new_region_10_120v_680c_120v_680c_opt.off |
| new_region_11_120v_380c | region (3) | 8.917945 | 1.8985107882 | infimum on the shape's edge (a sign condition is active): not attained | [0.4262441625, 0.2400881471, 0.2400881471, 0.0935795433] | [0.6001603745, 0.0335295528, 0.2990118908, 0.0672981819] (8.917915) | new_region_11_120v_380c_120v_380c_opt.off |
| new_wall_12_120v_440c | wall (2) | 14.552280 | 14.5522797588 | no progress: the optimiser found no feasible step from the seed (seed kept) | [0.0491045137, 0.3493222266, 0.5755263782, 0.0260468815] | same | new_wall_12_120v_440c_120v_440c_opt.off |
| snad | line (1) | 13.472194 | 1.2130146416 | minimum inside the shape (exact: b1 = b4 = 1 - 1/sqrt2, b2 = b3 = 1/sqrt2 - 1/2, R = sqrt((3 + sqrt2)/3); the optimiser had stopped at b1 = 0.2929009, R 1.2130618) | [0.2928932188, 0.2071067812, 0.2071067812, 0.2928932188] | same | snad_60v_90c_opt.off |

## Iodecaic

| shape | kind | R start | R best | status | best beta | off seed (R) | file |
|---|---|---|---|---|---|---|---|
| sosbid | region (3) | 25.463868 | 1.6760122298 | minimum inside the shape | [0.2464164258, 0.119725923, 0.3874412254, 0.2464164258] | same | sosbid_120v_380c_opt.off |
| new_region_01_120v_500c | region (3) | 15.006550 | 15.0065495408 | no progress: the optimiser found no feasible step from the seed (seed kept) | [0.0500721513, 0.2000594448, 0.7244752894, 0.0253931145] | same | new_region_01_120v_500c_120v_500c_opt.off |
| gosbid | region (3) | 19.772315 | 1.6739688188 | minimum inside the shape | [0.3150379619, 0.1849620381, 0.3150379619, 0.1849620381] | same | gosbid_120v_360c_opt.off |
| new_wall_01_120v_260c | wall (2) | 13.576008 | 4.5825756950 | infimum on the shape's edge (a sign condition is active): not attained | [0.1666666667, 0.1666666667, 0.6666666667, -0.0] | [0.1665678035, 0.1666469927, 0.6667208411, 6.43627e-05] (4.585537) | new_wall_01_120v_260c_120v_260c_opt.off |
| tetosbid | wall (2) | 6.748672 | 1.4421496953 | minimum inside the shape | [0.2739590735, 0.1614185982, 0.3552855831, 0.2093367452] | same | tetosbid_120v_180c_opt.off |
| petosbid | wall (2) | 4.718727 | 1.5411035007 | minimum inside the shape | [0.3636363636, 0.1818181818, 0.2727272727, 0.1818181818] | same | petosbid_120v_170c_opt.off |
| gippid | line (1) | 11.000000 | 1.0000000000 | minimum inside the shape | [0.25, 0.25, 0.25, 0.25] | same | gippid_120v_30c_opt.off |

## Contic

| shape | kind | R start | R best | status | best beta | off seed (R) | file |
|---|---|---|---|---|---|---|---|
| sabotic | region (3) | 46.307538 | 1.8866314650 | minimum inside the shape | [0.3267877261, 0.3267877261, 0.1732122739, 0.1732122739] | same | sabotic_2304v_1248c_opt.off |
| gabotic | region (3) | 3.866437 | 2.3065629649 | minimum inside the shape | [0.3487855803, 0.1512144197, 0.3487855803, 0.1512144197] | same | gabotic_2304v_1056c_opt.off |
| gabiparc | wall (2) | 47.713203 | 3.0000000000 | infimum on the shape's edge (a sign condition is active): not attained | [0.6, 0.0, 0.2, 0.2] | [0.6002162285, 0.0, 0.1998171051, 0.1999666663] (3.003828) | gabiparc_1152v_1248c_opt.off |
| mabiparc | wall (2) | 3.000000 | 1.8057908947 | minimum inside the shape | [0.3915773323, 0.0, 0.3915773323, 0.2168453354] | same | mabiparc_1152v_1056c_opt.off |
| sabiparc | wall (2) | 3.000000 | 1.4200457834 | minimum inside the shape | [0.3697947715, 0.3697947715, 0.0, 0.260410457] | same | sabiparc_1152v_1248c_opt.off |
| mobcotic | wall (2) | 3.919689 | 1.7071067812 | infimum on the shape's edge (a sign condition is active): not attained | [0.3431457505, 0.2426406871, 0.4142135624, -0.0] | [0.3429692678, 0.2427313816, 0.4142993506, 0.0] (1.708339) | mobcotic_1152v_864c_opt.off |
| sobcotic | wall (2) | 1.470127 | 1.2293022136 | minimum inside the shape | [0.3554330191, 0.3554330191, 0.2891339617, -0.0] | same | sobcotic_1152v_1248c_opt.off |
| gobcotic | wall (2) | 4.653010 | 4.4142135624 | infimum on the shape's edge (a sign condition is active): not attained | [0.1559037582, 0.1559037582, 0.6881924837, -0.0] | [0.1558952675, 0.1559145223, 0.6881902102, 0.0] (4.414439) | gobcotic_1152v_1056c_opt.off |
| pitbotic | wall (2) | 1.767767 | 1.7071067812 | minimum inside the shape | [0.3964466324, 0.160188585, 0.2734590197, 0.169905763] | same | pitbotic_2304v_384c_opt.off |
| catbotic | wall (2) | 1.735674 | 1.5411961001 | minimum inside the shape | [0.2273851233, 0.1762309769, 0.4488458134, 0.1475380864] | same | catbotic_2304v_528c_opt.off |
| bipec | line (1) | 19.485281 | 2.3065629649 | minimum inside the shape | [0.6975711606, 0.0, 0.0, 0.3024288394] | same | bipec_288v_528c_opt.off |
| bitec | line (1) | 5.828427 | 1.1278643263 | minimum inside the shape | [0.5, 0.5, 0.0, -0.0] | same | bitec_384v_1056c_opt.off |
| sabric | line (1) | 3.736189 | 2.0000000000 | infimum on the shape's edge (a sign condition is active): not attained | [0.6666666667, 0.0, 0.3333333333, 0.0] | [0.6667888632, 0.0, 0.3332111368, 0.0] (2.001100) | sabric_576v_1248c_opt.off |
| mabric | line (1) | 2.356048 | 1.3065629649 | minimum inside the shape | [0.5, 0.0, 0.5, 0.0] | same | mabric_576v_864c_opt.off |
| gabric | line (1) | 7.828427 | 3.4142135624 | infimum on the shape's edge (a sign condition is active): not attained | [0.2265409197, 0.0, 0.7734590803, 0.0] | [0.2264276486, 0.0, 0.7735723514, 0.0] (3.416422) | gabric_576v_1056c_opt.off |
| bimtec | line (1) | 3.000000 | 2.7071067812 | minimum inside the shape | [0.0, 0.2697521434, 0.7302478566, -0.0] | same | bimtec_576v_192c_opt.off |
| tispic | line (1) | 1.500000 | 1.4142135624 | minimum inside the shape (not unique: R is flat along the locus there) | [0.5857865445, 0.0, 0.1715730889, 0.2426403666] | same | tispic_1152v_384c_opt.off |
| tabcatic | line (1) | 10.010408 | 1.7071067812 | minimum inside the shape | [0.3431457505, 0.2426406871, 0.4142135624, -0.0] | same | tabcatic_1152v_384c_opt.off |
| gippic | line (1) | 7.828427 | 1.0000000000 | minimum inside the shape | [0.25, 0.25, 0.25, 0.25] | same | gippic_1152v_240c_opt.off |
| tabiparc | line (1) | 8.044318 | 1.5411961001 | minimum inside the shape | [0.3511533024, 0.0, 0.421002037, 0.2278446606] | same | tabiparc_1152v_528c_opt.off |
| ticont | line (1) | 2.207107 | 1.8477590650 | minimum inside the shape (not unique: R is flat along the locus there) | [0.1233086774, 0.2278445818, 0.6488467407, 0.0] | same | ticont_1152v_336c_opt.off |
| bicont | point (0) | 1.306563 | 1.3065629649 | point (no freedom) | [1.0, 0.0, 0.0, 0.0] | same | bicont_48v_288c_opt.off |
| spic | point (0) | 1.000000 | 1.0000000000 | point (no freedom) | [0.5, 0.0, 0.0, 0.5] | same | spic_144v_240c_opt.off |
| bamic | point (0) | 1.707107 | 1.7071067812 | point (no freedom) | [0.0, 0.0, 1.0, 0.0] | same | bamic_192v_192c_opt.off |
| cont | point (0) | 1.000000 | 1.0000000000 | point (no freedom) | [0.0, 0.5, 0.5, 0.0] | same | cont_288v_48c_opt.off |
| respic | point (0) | 1.414214 | 1.4142135624 | point (no freedom) | [0.6666666667, 0.0, 0.3333333333, 0.0] | same | respic_576v_384c_opt.off |
| recont | point (0) | 1.847759 | 1.8477590650 | point (no freedom) | [0.2265409197, 0.0, 0.7734590803, 0.0] | same | recont_576v_336c_opt.off |

## Kicontic

| shape | kind | R start | R best | status | best beta | off seed (R) | file |
|---|---|---|---|---|---|---|---|
| mabosc_or_gabosc_1 | region (3) | 20.032259 | 1.9784371515 | infimum on the shape's edge (a sign condition is active): not attained | [0.2697521434, -0.0, 0.4604957132, 0.2697521434] | [0.2693849922, 5.5953e-05, 0.4612287145, 0.2693303403] (1.983197) | mabosc_or_gabosc_1_1152v_5280c_opt.off |
| region_02_1152v_5280c | region (3) | 12.450867 | 7.0028104157 | infimum on the shape's edge (a sign condition is active): not attained | [0.1175875312, 0.1175875312, 0.7462977101, 0.0185272275] | [0.117532507, 0.11759513, 0.7463319287, 0.0185404343] (7.006224) | region_02_1152v_5280c_1152v_5280c_opt.off |
| region_03_1152v_5280c | region (3) | 12.024727 | 3.0747904739 | infimum on the shape's edge (a sign condition is active): not attained | [0.2414485547, 0.133330076, 0.4875095936, 0.1377117758] | [0.0707366895, 0.3042337364, 0.5884600174, 0.0365695568] (10.446791) | region_03_1152v_5280c_1152v_5280c_opt.off |
| apbosc | region (3) | 16.501498 | 2.3869165348 | infimum on the shape's edge (a sign condition is active): not attained | [0.3778059139, 0.1971515697, 0.3470665369, 0.0779759794] | [0.377490662, 0.1973604491, 0.347219156, 0.0779297329] (2.390615) | apbosc_1152v_3936c_opt.off |
| sabosc | region (3) | 23.958866 | 1.7023103291 | minimum inside the shape | [0.3998263062, 0.3116617804, 0.1434171563, 0.1450947571] | same | sabosc_1152v_4128c_opt.off |
| mabosc_or_gabosc_2 | region (3) | 17.046681 | 1.9514378726 | infimum on the shape's edge (a sign condition is active): not attained | [0.4360028985, 0.2288404697, 0.2288404697, 0.1063161621] | [0.4356337253, 0.2290652596, 0.2290653096, 0.1062357054] (1.954115) | mabosc_or_gabosc_2_1152v_5280c_opt.off |
| region_07_1152v_5280c | region (3) | 8.355230 | 2.5163282107 | infimum on the shape's edge (a sign condition is active): not attained | [0.5121011643, 0.0808762843, 0.2035112757, 0.2035112757] | [0.5120260589, 0.0808893931, 0.2034329738, 0.2036515741] (2.518001) | region_07_1152v_5280c_1152v_5280c_opt.off |
| region_08_1152v_5280c | region (3) | 11.957469 | 11.9574691469 | no progress: the optimiser found no feasible step from the seed (seed kept) | [0.6242398655, 0.0315738277, 0.0316202075, 0.3125660994] | same | region_08_1152v_5280c_1152v_5280c_opt.off |
| sigawbosc | wall (2) | 20.917544 | 1.9784371515 | infimum on the shape's edge (a sign condition is active): not attained | [0.2697521434, -0.0, 0.4604957132, 0.2697521434] | [0.2695123235, 3.17771e-05, 0.460943576, 0.2695123235] (1.981182) | sigawbosc_1152v_2688c_opt.off |
| wall_02_1152v_3552c | wall (2) | 12.509648 | 7.0028104157 | infimum on the shape's edge (a sign condition is active): not attained | [0.1175875312, 0.1175875312, 0.7462977101, 0.0185272275] | [0.1175327345, 0.1175761539, 0.7463475685, 0.0185435432] (7.006189) | wall_02_1152v_3552c_1152v_3552c_opt.off |
| wall_03_1152v_3552c | wall (2) | 10.496220 | 4.0129533058 | infimum on the shape's edge (a sign condition is active): not attained | [0.1752201314, 0.1752201314, 0.598238949, 0.0513207883] | [0.1751131338, 0.1753105339, 0.5982624354, 0.051313897] (4.015785) | wall_03_1152v_3552c_1152v_3552c_opt.off |
| catbosc | wall (2) | 7.885961 | 1.9075182508 | minimum inside the shape | [0.2878594395, 0.082507298, 0.4491655339, 0.1804677286] | same | catbosc_1152v_2256c_opt.off |
| petbosc | wall (2) | 28.256169 | 1.9979944526 | minimum inside the shape | [0.4264081109, 0.1450251917, 0.287891797, 0.1406749003] | same | petbosc_1152v_2112c_opt.off |
| gigawbosc | wall (2) | 17.027607 | 1.8057908947 | minimum inside the shape | [0.412073945, 0.2652084228, 0.2652084228, 0.0575092094] | same | gigawbosc_1152v_2976c_opt.off |
| wall_07_1152v_3552c | wall (2) | 8.793390 | 2.5163282107 | minimum inside the shape | [0.5121011643, 0.0808762843, 0.2035112757, 0.2035112757] | same | wall_07_1152v_3552c_1152v_3552c_opt.off |
| wall_08_1152v_3552c | wall (2) | 9.888103 | 5.8207546037 | infimum on the shape's edge (a sign condition is active): not attained | [0.7388835642, 0.0196665466, 0.1141504307, 0.1272994586] | [0.738751447, 0.0196890046, 0.1140738865, 0.1274856619] (5.822638) | wall_08_1152v_3552c_1152v_3552c_opt.off |
| region_09_1152v_6432c | region (3) | 16.635250 | 5.1699054209 | infimum on the shape's edge (a sign condition is active): not attained | [0.1559037582, 0.1559037582, 0.6881924837, 0.0] | [0.1558027158, 0.155841884, 0.6883510683, 4.3319e-06] (5.173950) | region_09_1152v_6432c_1152v_6432c_opt.off |
| region_10_1152v_6432c | region (3) | 6.524276 | 5.1699054209 | infimum on the shape's edge (a sign condition is active): not attained | [0.1559037582, 0.1559037582, 0.6881924837, -0.0] | [0.131929866, 0.1422133878, 0.71329694, 0.0125598062] (6.187795) | region_10_1152v_6432c_1152v_6432c_opt.off |
| region_11_1152v_6432c | region (3) | 18.974796 | 7.0028104157 | infimum on the shape's edge (a sign condition is active): not attained | [0.1175875312, 0.1175875312, 0.7462977101, 0.0185272275] | [0.1175121926, 0.1176291714, 0.7463287427, 0.0185298933] (7.007697) | region_11_1152v_6432c_1152v_6432c_opt.off |
| region_12_1152v_6432c | region (3) | 13.214097 | 3.1622776602 | infimum on the shape's edge (a sign condition is active): not attained | [0.6, -0.0, 0.2, 0.2] | [0.6002605123, 2.461e-06, 0.1998635311, 0.1998734957] (3.165446) | region_12_1152v_6432c_1152v_6432c_opt.off |
| region_13_1152v_6432c | region (3) | 5.185682 | 3.1623069967 | infimum on the shape's edge (a sign condition is active): not attained | [0.6000023013, 9.817e-07, 0.1999980114, 0.1999987056] | [0.6000539115, 2.29956e-05, 0.199913582, 0.2000095109] (3.163658) | region_13_1152v_6432c_1152v_6432c_opt.off |
| region_14_1152v_6432c | region (3) | 14.120967 | 5.8207546037 | infimum on the shape's edge (a sign condition is active): not attained | [0.7388835642, 0.0196665466, 0.1141504307, 0.1272994586] | [0.7389169525, 0.0196643364, 0.1140784824, 0.1273402286] (5.824376) | region_14_1152v_6432c_1152v_6432c_opt.off |
| snoc | line (1) | 14.372193 | 1.2062710560 | minimum inside the shape | [0.3042113339, 0.1957886661, 0.1957886661, 0.3042113339] | same | snoc_576v_816c_opt.off |

## Iocontic

| shape | kind | R start | R best | status | best beta | off seed (R) | file |
|---|---|---|---|---|---|---|---|
| sosbic | region (3) | 20.032259 | 1.7834317973 | minimum inside the shape | [0.2502557051, 0.0938215936, 0.4056669962, 0.2502557051] | same | sosbic_1152v_3552c_opt.off |
| gosbic | region (3) | 12.400556 | 1.7330188012 | minimum inside the shape | [0.3279754179, 0.195495494, 0.2985644918, 0.1779645962] | same | gosbic_1152v_3360c_opt.off |
| catosbic | wall (2) | 4.713161 | 1.4253578353 | minimum inside the shape | [0.2703702313, 0.1574609829, 0.3615851342, 0.2105836516] | same | catosbic_1152v_1680c_opt.off |
| petosbic | wall (2) | 1.983591 | 1.5742779762 | minimum inside the shape | [0.3887840056, 0.1610396079, 0.260686835, 0.1894895514] | same | petosbic_1152v_1536c_opt.off |
| region_03_1152v_4704c | region (3) | 8.691921 | 5.1699054209 | infimum on the shape's edge (a sign condition is active): not attained | [0.1559037582, 0.1559037582, 0.6881924837, 0.0] | [0.1558882608, 0.1558420756, 0.6882590947, 1.05689e-05] (5.172060) | region_03_1152v_4704c_1152v_4704c_opt.off |
| gippic | line (1) | 11.000000 | 1.0000000000 | minimum inside the shape | [0.25, 0.25, 0.25, 0.25] | same | gippic_1152v_240c_opt.off |

## Tessic

| shape | kind | R start | R best | status | best beta | off seed (R) | file |
|---|---|---|---|---|---|---|---|
| gidpith | region (3) | 1.000000 | 1.0000000000 | uniform: every edge the same length (R = 1), the seed kept | [0.25, 0.25, 0.25, 0.25] | same | gidpith_384v_80c_opt.off |
| tico | wall (2) | 1.000000 | 1.0000000000 | uniform: every edge the same length (R = 1), the seed kept | [0.3333333333, 0.3333333333, 0.3333333333, 0.0] | same | tico_192v_48c_opt.off |
| proh | wall (2) | 1.000000 | 1.0000000000 | uniform: every edge the same length (R = 1), the seed kept | [0.3333333333, 0.3333333333, 0.0, 0.3333333333] | same | proh_192v_80c_opt.off |
| prit | wall (2) | 1.000000 | 1.0000000000 | uniform: every edge the same length (R = 1), the seed kept | [0.3333333333, 0.0, 0.3333333333, 0.3333333333] | same | prit_192v_80c_opt.off |
| grit | wall (2) | 1.000000 | 1.0000000000 | uniform: every edge the same length (R = 1), the seed kept | [0.0, 0.3333333333, 0.3333333333, 0.3333333333] | same | grit_192v_56c_opt.off |
| thex | line (1) | 1.000000 | 1.0000000000 | uniform: every edge the same length (R = 1), the seed kept | [0.5, 0.5, 0.0, 0.0] | same | thex_48v_24c_opt.off |
| sidpith | line (1) | 1.000000 | 1.0000000000 | uniform: every edge the same length (R = 1), the seed kept | [0.5, 0.0, 0.0, 0.5] | same | sidpith_64v_80c_opt.off |
| tat | line (1) | 1.000000 | 1.0000000000 | uniform: every edge the same length (R = 1), the seed kept | [-0.0, 0.0, 0.5, 0.5] | same | tat_64v_24c_opt.off |
| rico | line (1) | 1.000000 | 1.0000000000 | uniform: every edge the same length (R = 1), the seed kept | [0.5, 0.0, 0.5, 0.0] | same | rico_96v_48c_opt.off |
| tah | line (1) | 1.000000 | 1.0000000000 | uniform: every edge the same length (R = 1), the seed kept | [0.0, 0.5, 0.5, 0.0] | same | tah_96v_24c_opt.off |
| srit | line (1) | 1.000000 | 1.0000000000 | uniform: every edge the same length (R = 1), the seed kept | [0.0, 0.5, 0.0, 0.5] | same | srit_96v_56c_opt.off |
| hex | point (0) | 1.000000 | 1.0000000000 | point (no freedom) | [1.0, 0.0, 0.0, 0.0] | same | hex_8v_16c_opt.off |
| tes | point (0) | 1.000000 | 1.0000000000 | point (no freedom) | [0.0, 0.0, 0.0, 1.0] | same | tes_16v_8c_opt.off |
| ico | point (0) | 1.000000 | 1.0000000000 | point (no freedom) | [0.0, 1.0, 0.0, 0.0] | same | ico_24v_24c_opt.off |
| rit | point (0) | 1.000000 | 1.0000000000 | point (no freedom) | [0.0, 0.0, 1.0, 0.0] | same | rit_32v_24c_opt.off |

## Kitessic

| shape | kind | R start | R best | status | best beta | off seed (R) | file |
|---|---|---|---|---|---|---|---|
| snet | region (3) | 20.836848 | 1.2500179158 | minimum inside the shape | [0.2994085643, 0.2163917, 0.1978824536, 0.2863172821] | same | snet_192v_272c_opt.off |

## Tessipyritic

| shape | kind | R start | R best | status | best beta | off seed (R) | file |
|---|---|---|---|---|---|---|---|
| pysnet | region (3) | 21.332799 | 1.4903232733 | minimum inside the shape | [0.3060083211, 0.2546345037, 0.1131125659, 0.3262446093] | same | pysnet_192v_176c_opt.off |
| sadi | wall (2) | 30.774316 | 1.0000000000 | minimum inside the shape | [0.3819660113, 0.2360679775, 0.3819660113, 0.0] | same | sadi_96v_144c_opt.off |

## Demitessic

| shape | kind | R start | R best | status | best beta | off seed (R) | file |
|---|---|---|---|---|---|---|---|
| tico | region (3) | 2.414214 | 1.0000000000 | minimum inside the shape | [0.3333333333, 0.3333333333, 0.3333333333, -0.0] | same | tico_192v_48c_opt.off |
| rico | wall (2) | 2.414214 | 1.0000000000 | minimum inside the shape | [0.5, -0.0, 0.5, -0.0] | same | rico_96v_48c_opt.off |
| tah | wall (2) | 2.414214 | 1.0000000000 | minimum inside the shape | [-0.0, 0.5, 0.5, 0.0] | same | tah_96v_24c_opt.off |
| rit | line (1) | 2.414214 | 1.0000000000 | minimum inside the shape | [0.0, 0.0, 1.0, -0.0] | same | rit_32v_24c_opt.off |
| thex | line (1) | 1.414214 | 1.0000000000 | minimum inside the shape | [0.0, 0.5857864376, 0.0, 0.4142135624] | same | thex_48v_24c_opt.off |
| hex | point (0) | 1.000000 | 1.0000000000 | point (no freedom) | [0.0, 0.0, 0.0, 1.0] | same | hex_8v_16c_opt.off |
| ico | point (0) | 1.000000 | 1.0000000000 | point (no freedom) | [0.0, 1.0, 0.0, 0.0] | same | ico_24v_24c_opt.off |

## Kidemitessic

| shape | kind | R start | R best | status | best beta | off seed (R) | file |
|---|---|---|---|---|---|---|---|
| sadi | region (3) | 29.819967 | 1.0000000000 | minimum inside the shape | [0.3819660113, 0.2360679775, 0.3819660113, -0.0] | same | sadi_96v_144c_opt.off |

## Icoic

contic_atlas/icoic_shapes.py ([3,4,3], CONTIC_SUB=ico). Sidpii, icoi and gidpii are most even where their two sets of 24 cells are equal: spic, cont and gippic.

| shape | kind | R seed | R best | status | best beta | file |
|---|---|---|---|---|---|---|
| ico | point (0) | 1.000000 | 1.0000000000 | point (no freedom) | [1.0, 0.0, 0.0, 0.0] | ico_24v_24c_opt.off |
| rico | point (0) | 1.000000 | 1.0000000000 | point (no freedom) | [0.0, 1.0, 0.0, 0.0] | rico_96v_48c_opt.off |
| sidpii | line (1) | 2.333333 | 1.0000000000 | minimum inside the shape | [0.5, 0.0, 0.0, 0.5] | sidpii_144v_240c_opt.off |
| tico | line (1) | 1.000000 | 1.0000000000 | minimum inside the shape | [0.5, 0.5, 0.0, 0.0] | tico_192v_48c_opt.off |
| srico | line (1) | 1.000000 | 1.0000000000 | minimum inside the shape | [0.5, 0.0, 0.5, 0.0] | srico_288v_144c_opt.off |
| icoi | line (1) | 1.857143 | 1.0000000000 | minimum inside the shape | [0.0, 0.5, 0.5, -0.0] | icoi_288v_48c_opt.off |
| grico | wall (2) | 1.000000 | 1.0000000000 | minimum inside the shape | [0.3333333333, 0.3333333333, 0.3333333333, -0.0] | grico_576v_144c_opt.off |
| prico | wall (2) | 1.000000 | 1.0000000000 | minimum inside the shape | [0.3333333333, 0.3333333333, 0.0, 0.3333333333] | prico_576v_240c_opt.off |
| gidpii | region (3) | 4.000000 | 1.0000000000 | minimum inside the shape | [0.25, 0.25, 0.25, 0.25] | gidpii_1152v_240c_opt.off |

## Hyic

hyic_atlas/hyic_shapes.py ([5,3,3]): one shape per face of the chamber; optimized and central seeds are the uniform ones.

| shape | kind | R seed | R best | status | best beta | file |
|---|---|---|---|---|---|---|
| hi | point (0) | 1.000000 | 1.0000000000 | uniform: every edge equal | [1.0, 0.0, 0.0, 0.0] | hi_600v_120c_opt.off |
| rahi | point (0) | 1.000000 | 1.0000000000 | uniform: every edge equal | [0.0, 1.0, 0.0, 0.0] | rahi_1200v_720c_opt.off |
| rox | point (0) | 1.000000 | 1.0000000000 | uniform: every edge equal | [0.0, 0.0, 1.0, 0.0] | rox_720v_720c_opt.off |
| ex | point (0) | 1.000000 | 1.0000000000 | uniform: every edge equal | [0.0, 0.0, 0.0, 1.0] | ex_120v_600c_opt.off |
| thi | line (1) | 1.500000 | 1.0000000000 | uniform: every edge equal | [0.5, 0.5, 0.0, 0.0] | thi_2400v_720c_opt.off |
| srahi | line (1) | 1.500000 | 1.0000000000 | uniform: every edge equal | [0.5, 0.0, 0.5, 0.0] | srahi_3600v_1920c_opt.off |
| sidpixhi | line (1) | 1.500000 | 1.0000000000 | uniform: every edge equal | [0.5, 0.0, 0.0, 0.5] | sidpixhi_2400v_2640c_opt.off |
| xhi | line (1) | 1.500000 | 1.0000000000 | uniform: every edge equal | [0.0, 0.5, 0.5, 0.0] | xhi_3600v_720c_opt.off |
| srix | line (1) | 1.500000 | 1.0000000000 | uniform: every edge equal | [0.0, 0.5, 0.0, 0.5] | srix_3600v_1440c_opt.off |
| tex | line (1) | 1.500000 | 1.0000000000 | uniform: every edge equal | [0.0, 0.0, 0.5, 0.5] | tex_1440v_720c_opt.off |
| grahi | wall (2) | 2.500000 | 1.0000000000 | uniform: every edge equal | [0.3333333333, 0.3333333333, 0.3333333333, 0.0] | grahi_7200v_1920c_opt.off |
| prix | wall (2) | 2.500000 | 1.0000000000 | uniform: every edge equal | [0.3333333333, 0.3333333333, 0.0, 0.3333333333] | prix_7200v_2640c_opt.off |
| prahi | wall (2) | 2.500000 | 1.0000000000 | uniform: every edge equal | [0.3333333333, 0.0, 0.3333333333, 0.3333333333] | prahi_7200v_2640c_opt.off |
| grix | wall (2) | 2.500000 | 1.0000000000 | uniform: every edge equal | [0.0, 0.3333333333, 0.3333333333, 0.3333333333] | grix_7200v_1440c_opt.off |
| gidpixhi | region (3) | 4.000000 | 1.0000000000 | uniform: every edge equal | [0.25, 0.25, 0.25, 0.25] | gidpixhi_14400v_2640c_opt.off |

## Pennic

Pennic [3,3,3] = S5 (order 120; domain: the whole chamber). The optimized seed of each is its uniform seed (R = 1), reached from an uneven start; the central seed (central_off/pennic) is the same point (decaic_atlas/pennic_shapes.py).

| shape | kind | optimized = central beta | vertices, cells |
|---|---|---|---|
| pen | point (0) | [1.0, 0.0, 0.0, 0.0] (same: True) | [5, 5] |
| rap | point (0) | [0.0, 1.0, 0.0, 0.0] (same: True) | [10, 10] |
| tip | line (1) | [0.5, 0.5, 0.0, 0.0] (same: True) | [20, 10] |
| srip | line (1) | [0.5, 0.0, 0.5, 0.0] (same: True) | [30, 20] |
| grip | wall (2) | [0.333333, 0.333333, 0.333333, 0.0] (same: True) | [60, 20] |
| prip | wall (2) | [0.333333, 0.333333, 0.0, 0.333333] (same: True) | [60, 30] |

## Kipennic (chiral examples)

Kipennic [3,3,3]+ = A5 (order 60). Every seed inside the chamber gives the same polytope combinatorially (6000 points, one arrangement key): the kipennic snad family (60 vertices: 5+5 snits, 10+10 gyro-traps, 60 tets; three degrees of freedom). So the six pennic shapes have no chiral versions of their own: each example is the parent's uniform seed pushed off its mirrors (the zero betas raised to 0.06, 0.1, 0.14), a snad of this family that still resembles the parent, with symmetry exactly 60 (only the rotations). Not optimized: the most even member is snad itself, on b1 = b4, b2 = b3, where it gains the swap.

| example | seed beta | vertices, cells | symmetry | edge ratio | margin | file |
|---|---|---|---|---|---|---|
| pen_chiral | [0.769231, 0.046154, 0.076923, 0.107692] | 60, 90 | 60 | 7.366 | 5.3e-02 | pen_chiral_60v_90c.off |
| rap_chiral | [0.046154, 0.769231, 0.076923, 0.107692] | 60, 90 | 60 | 9.034 | 5.3e-02 | rap_chiral_60v_90c.off |
| tip_chiral | [0.431034, 0.431034, 0.051724, 0.086207] | 60, 90 | 60 | 6.186 | 5.6e-02 | tip_chiral_60v_90c.off |
| srip_chiral | [0.431034, 0.051724, 0.431034, 0.086207] | 60, 90 | 60 | 6.063 | 5.8e-02 | srip_chiral_60v_90c.off |
| grip_chiral | [0.314465, 0.314465, 0.314465, 0.056604] | 60, 90 | 60 | 1.705 | 7.5e-02 | grip_chiral_60v_90c.off |
| prip_chiral | [0.314465, 0.314465, 0.056604, 0.314465] | 60, 90 | 60 | 1.705 | 6.8e-02 | prip_chiral_60v_90c.off |

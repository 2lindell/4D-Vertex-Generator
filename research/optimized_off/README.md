# Optimized seeds and .off files (decaic, kidecaic, iodecaic)

The most even seed of each shape: the one that minimises R = longest edge / shortest edge (decaic_atlas/optimize_seeds.py, as swirlprism_atlas/exact_seeds.py). Edge lengths squared are exact quadratic forms in beta; the minimum is found on the shape's locus (its mirror / wall equations) while every sign condition of the shape holds, then solved at 40 digits where the active conditions allow (column 'exact').

* **minimum inside the shape**: attained; R rises at every point of a ring of small steps round it on the locus ('not unique': R is flat along some direction there, a curve of equally even seeds).
* **infimum on the shape's edge**: R falls all the way to the shape's boundary (often a mirror, where a snub becomes the decaic shape): never attained by the shape itself. Its .off uses a seed just inside (column 'off seed', with its R).
* **no progress**: no feasible step from the seed was found; the seed is kept.
* Kidecaic and iodecaic shapes on the mirrors are the decaic shapes (same orbits): their files are the decaic ones.

Seeds: beta (beta_k = x_k - x_{k+1}); vertices +-x[perm] for the group (decaic all; kidecaic even perms, both signs; iodecaic even perms +, odd perms -), x = the seed vector in each JSON (optimized_seeds*.json).

## Decaic

| shape | kind | R start | R best | status | exact | best beta | off seed (R) | file |
|---|---|---|---|---|---|---|---|---|
| sabotid | region (3) | 37.652747 | 1.8451542547 | minimum inside the shape | 40 | [0.324262604, 0.324262604, 0.175737396, 0.175737396] | same | sabotid_240v_140c_opt.off |
| gabotid | region (3) | 3.897613 | 2.2909944487 | minimum inside the shape | 40 | [0.348070239, 0.151929761, 0.348070239, 0.151929761] | same | gabotid_240v_120c_opt.off |
| gabipard | wall (2) | 19.000000 | 3.0000000000 | infimum on the shape's edge (a sign condition is active): not attained | 40 | [0.6, 0.0, 0.2, 0.2] | [0.6001916674, 0.0, 0.1998416661, 0.1999666665] (3.003336) | gabipard_120v_140c_opt.off |
| mobcated | wall (2) | 3.000000 | 1.7071067812 | infimum on the shape's edge (a sign condition is active): not attained | 40 | [0.3431457505, 0.2426406871, 0.4142135624, 0.0] | [0.3431359381, 0.2425647129, 0.4142993491, 0.0] (1.707995) | mobcated_120v_100c_opt.off |
| sobcated | wall (2) | 1.732051 | 1.2110322250 | minimum inside the shape | 40 | [0.3538893679, 0.3538893679, 0.2922212643, 0.0] | same | sobcated_120v_140c_opt.off |
| mabipard | wall (2) | 3.000000 | 1.7207592201 | minimum inside the shape | 40 | [0.3874258867, 0.0, 0.3874258867, 0.2251482266] | same | mabipard_120v_120c_opt.off |
| sabipard | wall (2) | 3.500000 | 1.5811388301 | minimum inside the shape | - | [0.3071033604, 0.0, 0.3857932791, 0.3071033604] | same | sabipard_120v_140c_opt.off |
| gobcated | wall (2) | 9.000000 | 4.0000000000 | infimum on the shape's edge (a sign condition is active): not attained | 40 | [0.1666666667, 0.1666666667, 0.6666666667, 0.0] | [0.166583333, 0.1666666667, 0.6667500003, 0.0] (4.002501) | gobcated_120v_120c_opt.off |
| petbotid | wall (2) | 1.767767 | 1.7071067812 | minimum inside the shape | 40 | [0.3963146941, 0.1603926828, 0.2738074364, 0.1694851868] | same | petbotid_240v_50c_opt.off |
| tetbotid | wall (2) | 1.732051 | 1.5773502692 | minimum inside the shape | 40 | [0.2376286257, 0.1753932606, 0.4363276016, 0.1506505121] | same | tetbotid_240v_60c_opt.off |
| bited | line (1) | 5.000000 | 1.1832159566 | minimum inside the shape | 40 | [0.5, 0.5, 0.0, 0.0] | same | bited_40v_120c_opt.off |
| biped | line (1) | 13.000000 | 2.2909944487 | minimum inside the shape | 40 | [0.696140478, 0.0, 0.0, 0.303859522] | same | biped_40v_60c_opt.off |
| bimted | line (1) | 3.000000 | 2.5811388301 | minimum inside the shape | 40 | [-0.0, 0.2792407799, 0.7207592201, 0.0] | same | bimted_60v_30c_opt.off |
| gabred | line (1) | 7.000000 | 3.0000000000 | infimum on the shape's edge (a sign condition is active): not attained | 40 | [0.25, 0.0, 0.75, 0.0] | [0.2498749995, 0.0, 0.7501250005, 0.0] (3.002001) | gabred_60v_120c_opt.off |
| mabred | line (1) | 3.363636 | 2.0000000000 | infimum on the shape's edge (a sign condition is active): not attained | 40 | [0.6666666667, 0.0, 0.3333333333, 0.0] | [0.6667708345, 0.0, 0.3332291655, 0.0] (2.000938) | mabred_60v_140c_opt.off |
| sabred | line (1) | 1.426935 | 1.2909944487 | minimum inside the shape | 40 | [0.5, 0.0, 0.5, 0.0] | same | sabred_60v_100c_opt.off |
| tabipard | line (1) | 7.505553 | 1.5773502692 | minimum inside the shape | 40 | [0.3660254038, 0.0, 0.4019237886, 0.2320508076] | same | tabipard_120v_60c_opt.off |
| gippid | line (1) | 11.000000 | 1.0000000000 | minimum inside the shape | 40 | [0.25, 0.25, 0.25, 0.25] | same | gippid_120v_30c_opt.off |
| tadeca | line (1) | 2.000000 | 1.7320508076 | minimum inside the shape (not unique: R is flat along the locus there) | 40 | [0.1342942076, 0.2314115848, 0.6342942076, 0.0] | same | tadeca_120v_40c_opt.off |
| tabcated | line (1) | 8.838835 | 1.7071067812 | minimum inside the shape | 40 | [0.3431457505, 0.2426406871, 0.4142135624, 0.0] | same | tabcated_120v_50c_opt.off |
| tispid | line (1) | 1.500000 | 1.4142135624 | minimum inside the shape (not unique: R is flat along the locus there) | 40 | [0.5857869049, 0.0, 0.1715738097, 0.2426392854] | same | tispid_120v_50c_opt.off |
| bideca | point (0) | 1.290994 | 1.2909944487 | point (no freedom) | - | [1.0, 0.0, 0.0, 0.0] | same | bideca_10v_30c_opt.off |
| spid | point (0) | 1.000000 | 1.0000000000 | point (no freedom) | - | [0.5, 0.0, 0.0, 0.5] | same | spid_20v_30c_opt.off |
| apid | point (0) | 1.581139 | 1.5811388301 | point (no freedom) | - | [0.0, 0.0, 1.0, 0.0] | same | apid_20v_30c_opt.off |
| deca | point (0) | 1.000000 | 1.0000000000 | point (no freedom) | - | [0.0, 0.5, 0.5, 0.0] | same | deca_30v_10c_opt.off |
| redeca | point (0) | 1.732051 | 1.7320508076 | point (no freedom) | - | [0.25, 0.0, 0.75, 0.0] | same | redeca_60v_40c_opt.off |
| respid | point (0) | 1.414214 | 1.4142135624 | point (no freedom) | - | [0.6666666667, 0.0, 0.3333333333, 0.0] | same | respid_60v_50c_opt.off |

## Kidecaic

| shape | kind | R start | R best | status | exact | best beta | off seed (R) | file |
|---|---|---|---|---|---|---|---|---|
| mebosid | region (3) | 25.463868 | 1.8708286934 | infimum on the shape's edge (a sign condition is active): not attained | 40 | [0.2792407799, -0.0, 0.4415184401, 0.2792407799] | [0.2788728971, 4.05602e-05, 0.4422532627, 0.27883328] (1.875087) | mebosid_120v_560c_opt.off |
| new_region_01_120v_560c | region (3) | 15.840267 | 6.3256565248 | infimum on the shape's edge (a sign condition is active): not attained | 40 | [0.1245463961, 0.1245463961, 0.7296479065, 0.0212593014] | [0.0901207565, 0.1017394826, 0.7849696088, 0.0231701521] (9.032302) | new_region_01_120v_560c_120v_560c_opt.off |
| new_region_02_120v_680c | region (3) | 15.306752 | 6.3256565248 | infimum on the shape's edge (a sign condition is active): not attained | 40 | [0.1245463961, 0.1245463961, 0.7296479065, 0.0212593014] | [0.1244719184, 0.1245719694, 0.7296926768, 0.0212634354] (6.329775) | new_region_02_120v_680c_120v_680c_opt.off |
| new_region_03_120v_680c | region (3) | 14.469616 | 4.5825756950 | infimum on the shape's edge (a sign condition is active): not attained | 40 | [0.1666666667, 0.1666666667, 0.6666666667, 0.0] | [0.1135314351, 0.23878307, 0.6361131751, 0.0115723198] (6.863798) | new_region_03_120v_680c_120v_680c_opt.off |
| new_region_04_120v_560c | region (3) | 14.401088 | 2.7165684733 | infimum on the shape's edge (a sign condition is active): not attained | - | [0.2665823566, 0.1205963407, 0.4590602489, 0.1537610538] | [0.095644216, 0.3016208508, 0.5503223439, 0.0524125893] (6.860303) | new_region_04_120v_560c_120v_560c_opt.off |
| apbosid | region (3) | 41.702218 | 2.2671357138 | infimum on the shape's edge (a sign condition is active): not attained | 40 | [0.4394656318, 0.0, 0.3445462982, 0.21598807] | [0.4390762378, 0.0004497643, 0.3446765232, 0.2157974747] (2.269071) | apbosid_120v_420c_opt.off |
| sabosid | region (3) | 27.302256 | 1.6652318805 | minimum inside the shape | 40 | [0.3962831202, 0.3101654558, 0.1489627514, 0.1445886726] | same | sabosid_120v_440c_opt.off |
| gabosid | region (3) | 20.286319 | 1.8985107882 | infimum on the shape's edge (a sign condition is active): not attained | 40 | [0.4262441625, 0.2400881471, 0.2400881471, 0.0935795433] | [0.4258745808, 0.2403087959, 0.2403088349, 0.0935077884] (1.901003) | gabosid_120v_560c_opt.off |
| new_region_05_120v_560c | region (3) | 10.317586 | 2.5155111470 | infimum on the shape's edge (a sign condition is active): not attained | 40 | [0.5120065698, 0.0809138827, 0.2035397738, 0.2035397738] | [0.5119191832, 0.0809331484, 0.2034614118, 0.2036862566] (2.517293) | new_region_05_120v_560c_120v_560c_opt.off |
| new_region_06_120v_560c | region (3) | 15.663981 | 1.8985107882 | infimum on the shape's edge (a sign condition is active): not attained | 40 | [0.4262441625, 0.2400881471, 0.2400881471, 0.0935795433] | [0.5708700541, 0.0304568462, 0.0304930643, 0.3681800353] (12.869397) | new_region_06_120v_560c_120v_560c_opt.off |
| new_region_07_120v_680c | region (3) | 10.099666 | 3.1622776602 | infimum on the shape's edge (a sign condition is active): not attained | 40 | [0.6, -0.0, 0.2, 0.2] | [0.5751686008, 0.0245567534, 0.0544801304, 0.3457945154] (9.578563) | new_region_07_120v_680c_120v_680c_opt.off |
| new_region_08_120v_680c | region (3) | 10.401016 | 5.4848991540 | infimum on the shape's edge (a sign condition is active): not attained | 40 | [0.7238222469, 0.0222340399, 0.1216464783, 0.1322972348] | [0.6302684585, 0.0250873021, 0.0544924632, 0.2901517762] (9.846168) | new_region_08_120v_680c_120v_680c_opt.off |
| new_region_09_120v_680c | region (3) | 5.851483 | 3.1622950806 | infimum on the shape's edge (a sign condition is active): not attained | - | [0.6000012617, 6.882e-07, 0.1999988524, 0.1999991977] | [0.6000256547, 2.59567e-05, 0.1998990578, 0.2000493307] (3.163869) | new_region_09_120v_680c_120v_680c_opt.off |
| sigwebosid | wall (2) | 26.398538 | 1.8708286934 | infimum on the shape's edge (a sign condition is active): not attained | 40 | [0.2792407799, -0.0, 0.4415184401, 0.2792407799] | [0.2789856633, 2.53719e-05, 0.4420033016, 0.2789856633] (1.873520) | sigwebosid_120v_290c_opt.off |
| new_wall_01_120v_380c | wall (2) | 15.904112 | 6.3256565248 | infimum on the shape's edge (a sign condition is active): not attained | 40 | [0.1245463961, 0.1245463961, 0.7296479065, 0.0212593014] | [0.1244737874, 0.1244851692, 0.7297611927, 0.0212798506] (6.329660) | new_wall_01_120v_380c_120v_380c_opt.off |
| new_wall_02_120v_440c | wall (2) | 12.750220 | 6.3256565248 | infimum on the shape's edge (a sign condition is active): not attained | 40 | [0.1245463961, 0.1245463961, 0.7296479065, 0.0212593014] | [0.124484393, 0.1245450951, 0.7296988088, 0.0212717031] (6.329006) | new_wall_02_120v_440c_120v_440c_opt.off |
| new_wall_03_120v_500c | wall (2) | 11.362957 | 6.3256565248 | infimum on the shape's edge (a sign condition is active): not attained | 40 | [0.1245463961, 0.1245463961, 0.7296479065, 0.0212593014] | [0.1244863582, 0.124669552, 0.729571806, 0.0212722838] (6.328511) | new_wall_03_120v_500c_120v_500c_opt.off |
| new_wall_04_120v_500c | wall (2) | 11.503736 | 4.5825756950 | infimum on the shape's edge (a sign condition is active): not attained | 40 | [0.1666666667, 0.1666666667, 0.6666666667, 0.0] | [0.1665628116, 0.1668113246, 0.6666100888, 1.57749e-05] (4.585668) | new_wall_04_120v_500c_120v_500c_opt.off |
| tetbosid | wall (2) | 11.692719 | 1.9139484184 | minimum inside the shape | 40 | [0.2968602152, 0.0871919302, 0.4350354939, 0.1809123607] | same | tetbosid_120v_240c_opt.off |
| petbosid | wall (2) | 33.873448 | 1.9834008275 | minimum inside the shape | 40 | [0.4149782763, 0.1566487331, 0.2999027519, 0.1284702387] | same | petbosid_120v_230c_opt.off |
| gigwebosid | wall (2) | 20.272881 | 1.7207592201 | minimum inside the shape | 40 | [0.4007450022, 0.2833695086, 0.2833695086, 0.0325159806] | same | gigwebosid_120v_320c_opt.off |
| new_wall_05_120v_380c | wall (2) | 9.715348 | 3.4205262753 | infimum on the shape's edge (a sign condition is active): not attained | 40 | [0.1875, 0.1875, 0.5625, 0.0625] | [0.1873831933, 0.1875763659, 0.5625606766, 0.0624797642] (3.423122) | new_wall_05_120v_380c_120v_380c_opt.off |
| new_wall_06_120v_380c | wall (2) | 10.781103 | 2.5155111470 | minimum inside the shape | 40 | [0.5120065698, 0.0809138827, 0.2035397738, 0.2035397738] | same | new_wall_06_120v_380c_120v_380c_opt.off |
| new_wall_07_120v_380c | wall (2) | 6.370752 | 5.4848991540 | infimum on the shape's edge (a sign condition is active): not attained | 40 | [0.7238222469, 0.0222340399, 0.1216464783, 0.1322972348] | [0.7238213868, 0.0222383484, 0.1216250425, 0.1323152223] (5.485675) | new_wall_07_120v_380c_120v_380c_opt.off |
| new_wall_08_120v_440c | wall (2) | 10.058360 | 3.1622776602 | infimum on the shape's edge (a sign condition is active): not attained | 40 | [0.6, -0.0, 0.2, 0.2] | [0.5999727462, 1.80112e-05, 0.1998495238, 0.2001597187] (3.164640) | new_wall_08_120v_440c_120v_440c_opt.off |
| new_wall_09_120v_440c | wall (2) | 12.298535 | 5.4848991540 | infimum on the shape's edge (a sign condition is active): not attained | 40 | [0.7238222469, 0.0222340399, 0.1216464783, 0.1322972348] | [0.7236551003, 0.0222617295, 0.1215611942, 0.1325219759] (5.486726) | new_wall_09_120v_440c_120v_440c_opt.off |
| new_wall_10_120v_500c | wall (2) | 10.338378 | 5.4848991540 | infimum on the shape's edge (a sign condition is active): not attained | 40 | [0.7238222469, 0.0222340399, 0.1216464783, 0.1322972348] | [0.7237054183, 0.0222529486, 0.1215702819, 0.1324713512] (5.486881) | new_wall_10_120v_500c_120v_500c_opt.off |
| new_wall_11_120v_500c | wall (2) | 6.120185 | 3.1622776602 | infimum on the shape's edge (a sign condition is active): not attained | 40 | [0.6, -0.0, 0.2, 0.2] | [0.6000268786, 2.27109e-05, 0.199894808, 0.2000556026] (3.163978) | new_wall_11_120v_500c_120v_500c_opt.off |
| snad | line (1) | 13.472194 | 1.2130618279 | minimum inside the shape | - | [0.2929009139, 0.2070990861, 0.2070990861, 0.2929009139] | same | snad_60v_90c_opt.off |

## Iodecaic

| shape | kind | R start | R best | status | exact | best beta | off seed (R) | file |
|---|---|---|---|---|---|---|---|---|
| sosbid | region (3) | 25.463868 | 1.6760122298 | minimum inside the shape | - | [0.2464164258, 0.119725923, 0.3874412254, 0.2464164258] | same | sosbid_120v_380c_opt.off |
| new_region_01_120v_500c | region (3) | 15.006550 | 15.0065495408 | no progress: the optimiser found no feasible step from the seed (seed kept) | - | [0.0500721513, 0.2000594448, 0.7244752894, 0.0253931145] | same | new_region_01_120v_500c_120v_500c_opt.off |
| gosbid | region (3) | 19.772315 | 1.6739688188 | minimum inside the shape | 40 | [0.3150379619, 0.1849620381, 0.3150379619, 0.1849620381] | same | gosbid_120v_360c_opt.off |
| new_wall_01_120v_260c | wall (2) | 13.576008 | 4.5825756950 | infimum on the shape's edge (a sign condition is active): not attained | 40 | [0.1666666667, 0.1666666667, 0.6666666667, -0.0] | [0.1665678035, 0.1666469927, 0.6667208411, 6.43627e-05] (4.585537) | new_wall_01_120v_260c_120v_260c_opt.off |
| tetosbid | wall (2) | 6.748672 | 1.4421496953 | minimum inside the shape | 40 | [0.2739590735, 0.1614185982, 0.3552855831, 0.2093367452] | same | tetosbid_120v_180c_opt.off |
| petosbid | wall (2) | 4.718727 | 1.5411035007 | minimum inside the shape | 40 | [0.3636363636, 0.1818181818, 0.2727272727, 0.1818181818] | same | petosbid_120v_170c_opt.off |
| gippid | line (1) | 11.000000 | 1.0000000000 | minimum inside the shape | 40 | [0.25, 0.25, 0.25, 0.25] | same | gippid_120v_30c_opt.off |


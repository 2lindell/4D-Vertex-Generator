# Central seeds and .off files

The central seed of a shape is the point of its locus **deepest inside it**: as far as possible from every place the polytope changes. Where the optimized seed (optimized_off/) makes the edges as even as possible and often sits on or near the shape's edge, the central seed keeps every cell clearly formed.

How (decaic_atlas/central_seeds.py; contic_atlas/central_seeds.py runs it on the contic groups):

* the shape is its locus (the mirror / wall / line equations it lies on) where every sign condition of its polychoron keeps its sign (the forms where a vertex reaches a cell's 3-flat), inside the domain (beta >= 0, b1 >= b4);
* the depth of a seed is its least distance to any of those boundaries, each to first order (f / |grad f| along the locus, beta summing to 1);
* the central seed maximises the depth (SLSQP from the map's seed, the optimized seed and their midpoint) and is kept only if it is still the shape: the same arrangement key, and the polychoron rebuilt with the shape's own vertex and cell counts. A point (no freedom) is its own centre.

This replaces an earlier version of this folder that interpolated 35% of the way from the map's seed to the optimized seed: that did not check the result was still the shape (several decaic files were other polytopes, e.g. bimted with 33 cells, biped 79, gabipard 144) and its contic-family files had no hull (0 cells).

Groups: decaic [[3,3,3]] (order 240), kidecaic [[3,3,3]]+ and iodecaic [[3,3,3]+] (120 each); contic [[3,4,3]] (the 24-cell / 48-cell family, order 2304), kicontic [[3,4,3]]+ and iocontic [[3,4,3]+] (1152 each). Seed vectors: R^5 (decaic family, vertices sign * x[perm]) or R^4 (contic family, the standard F4 frame).

| group | files | moved deeper | kept the seed |
|---|---|---|---|
| decaic | 27 | 21 | - |
| kidecaic | 32 | 31 | new_region_11_120v_380c (points: their own centre) |
| iodecaic | 7 | 7 | - |
| contic | 27 | 21 | - |
| kicontic | 23 | 22 | region_08_1152v_5280c (points: their own centre) |
| iocontic | 6 | 6 | - |

## decaic

| shape | kind | depth (seed -> centre) | central beta | file |
|---|---|---|---|---|
| sabotid | region (3) | 1.11e-02 -> 9.07e-02 | [0.3592875069, 0.3046669146, 0.1712593231, 0.1647862554] | sabotid_240v_140c_central.off |
| gabotid | region (3) | 1.91e-02 -> 4.95e-02 | [0.3516174694, 0.129470247, 0.3860346721, 0.1328776115] | gabotid_240v_120c_central.off |
| gabipard | wall (2) | 2.46e-02 -> 9.88e-02 | [0.6928265094, 0.0, 0.1752686698, 0.1319048209] | gabipard_120v_140c_central.off |
| mobcated | wall (2) | 8.75e-02 -> 9.55e-02 | [0.2841274313, 0.1952602567, 0.520612312, 0.0] | mobcated_120v_100c_central.off |
| sobcated | wall (2) | 5.63e-02 -> 1.41e-01 | [0.3635334168, 0.4187814366, 0.2176851466, 0.0] | sobcated_120v_140c_central.off |
| mabipard | wall (2) | 7.72e-02 -> 8.86e-02 | [0.4755786074, 0.0, 0.3292053558, 0.1952160369] | mabipard_120v_120c_central.off |
| sabipard | wall (2) | 5.81e-02 -> 6.16e-02 | [0.204460936, 0.0, 0.6782136245, 0.1173254395] | sabipard_120v_140c_central.off |
| gobcated | wall (2) | 5.36e-02 -> 7.94e-02 | [0.1220949448, 0.1060264603, 0.7718785949, 0.0] | gobcated_120v_120c_central.off |
| petbotid | wall (2) | 1.34e-01 -> 1.49e-01 | [0.3881283465, 0.175108642, 0.3015826189, 0.1351803926] | petbotid_240v_50c_central.off |
| tetbotid | wall (2) | 1.32e-01 -> 1.49e-01 | [0.2637363014, 0.1306149838, 0.4475611284, 0.1580875865] | tetbotid_240v_60c_central.off |
| bited | line (1) | 1.38e-01 -> 4.06e-01 | [0.3547774564, 0.6452225436, 0.0, 0.0] | bited_40v_120c_central.off |
| biped | line (1) | 3.46e-02 -> 2.02e-01 | [0.6482675827, 0.0, 0.0, 0.3517324173] | biped_40v_60c_central.off |
| bimted | line (1) | 2.12e-01 -> 2.84e-01 | [0.0, 0.3110177635, 0.6889822365, 0.0] | bimted_60v_30c_central.off |
| gabred | line (1) | 1.09e-01 -> 1.25e-01 | [0.1390001239, 0.0, 0.8609998761, 0.0] | gabred_60v_120c_central.off |
| mabred | line (1) | 1.20e-01 -> 1.33e-01 | [0.75, 0.0, 0.25, 0.0] | mabred_60v_140c_central.off |
| sabred | line (1) | 1.93e-01 -> 2.07e-01 | [0.4725869669, 0.0, 0.5274130331, 0.0] | sabred_60v_100c_central.off |
| tabipard | line (1) | 6.72e-02 -> 2.98e-01 | [0.3554551648, 0.0, 0.4336345055, 0.2109103297] | tabipard_120v_60c_central.off |
| gippid | line (1) | 4.37e-02 -> 4.29e-01 | [0.2363988743, 0.2636011257, 0.2636011257, 0.2363988743] | gippid_120v_30c_central.off |
| tadeca | line (1) | 9.57e-02 -> 1.43e-01 | [0.1906437133, 0.1187125735, 0.6906437133, 0.0] | tadeca_120v_40c_central.off |
| tabcated | line (1) | 5.36e-02 -> 4.25e-01 | [0.3333333333, 0.25, 0.4166666667, 0.0] | tabcated_120v_50c_central.off |
| tispid | line (1) | 9.92e-02 -> 1.48e-01 | [0.6270083225, 0.0, 0.2540166451, 0.1189750324] | tispid_120v_50c_central.off |
| bideca | point (0) | - | [1.0, 0.0, 0.0, 0.0] | bideca_10v_30c_central.off |
| spid | point (0) | - | [0.5, 0.0, 0.0, 0.5] | spid_20v_30c_central.off |
| apid | point (0) | - | [0.0, 0.0, 1.0, 0.0] | apid_20v_30c_central.off |
| deca | point (0) | - | [0.0, 0.5, 0.5, 0.0] | deca_30v_10c_central.off |
| redeca | point (0) | - | [0.25, 0.0, 0.75, 0.0] | redeca_60v_40c_central.off |
| respid | point (0) | - | [0.6666666667, 0.0, 0.3333333333, 0.0] | respid_60v_50c_central.off |

## kidecaic

| shape | kind | depth (seed -> centre) | central beta | file |
|---|---|---|---|---|
| mebosid | region (3) | 1.60e-02 -> 4.86e-02 | [0.1876997586, 0.0645092919, 0.6321215663, 0.1156693832] | mebosid_120v_560c_central.off |
| new_region_01_120v_560c | region (3) | 2.36e-03 -> 7.84e-03 | [0.054794966, 0.1911053487, 0.7182029116, 0.0358967737] | new_region_01_120v_560c_120v_560c_central.off |
| new_region_02_120v_680c | region (3) | 1.01e-04 -> 6.88e-03 | [0.0563467219, 0.1375145445, 0.786862571, 0.0192761626] | new_region_02_120v_680c_120v_680c_central.off |
| new_region_03_120v_680c | region (3) | 1.64e-03 -> 1.61e-02 | [0.0774247414, 0.2387023079, 0.6655936716, 0.0182792791] | new_region_03_120v_680c_120v_680c_central.off |
| new_region_04_120v_560c | region (3) | 1.94e-04 -> 1.72e-02 | [0.1087755144, 0.2499462144, 0.6088556332, 0.0324226381] | new_region_04_120v_560c_120v_560c_central.off |
| apbosid | region (3) | 6.95e-03 -> 5.83e-02 | [0.3759063582, 0.1024244103, 0.4177712945, 0.103897937] | apbosid_120v_420c_central.off |
| sabosid | region (3) | 1.03e-02 -> 1.12e-01 | [0.3587339317, 0.3589101064, 0.1556993469, 0.126656615] | sabosid_120v_440c_central.off |
| gabosid | region (3) | 2.62e-05 -> 6.02e-02 | [0.5748609202, 0.131782834, 0.2246201012, 0.0687361447] | gabosid_120v_560c_central.off |
| new_region_05_120v_560c | region (3) | 3.09e-03 -> 2.35e-02 | [0.5733076114, 0.0428203352, 0.1453833995, 0.2384886539] | new_region_05_120v_560c_120v_560c_central.off |
| new_region_06_120v_560c | region (3) | 2.62e-05 -> 6.02e-03 | [0.7212742195, 0.0303944045, 0.0463606171, 0.2019707589] | new_region_06_120v_560c_120v_560c_central.off |
| new_region_07_120v_680c | region (3) | 3.10e-04 -> 1.60e-02 | [0.668614158, 0.0158153597, 0.073720767, 0.2418497152] | new_region_07_120v_680c_120v_680c_central.off |
| new_region_08_120v_680c | region (3) | 9.84e-04 -> 5.65e-03 | [0.7832737605, 0.017076951, 0.05020091, 0.1494483785] | new_region_08_120v_680c_120v_680c_central.off |
| new_region_09_120v_680c | region (3) | 1.22e-03 -> 2.05e-03 | [0.5897002876, 0.0211425666, 0.0908817438, 0.298275402] | new_region_09_120v_680c_120v_680c_central.off |
| sigwebosid | wall (2) | 2.50e-02 -> 1.15e-01 | [0.1714766376, 0.1108208752, 0.5462258497, 0.1714766376] | sigwebosid_120v_290c_central.off |
| new_wall_01_120v_380c | wall (2) | 2.16e-02 -> 3.93e-02 | [0.0749199034, 0.0914743279, 0.8050128481, 0.0285929206] | new_wall_01_120v_380c_120v_380c_central.off |
| new_wall_02_120v_440c | wall (2) | 3.77e-02 -> 3.95e-02 | [0.0656862875, 0.1351116929, 0.7726540723, 0.0265479473] | new_wall_02_120v_440c_120v_440c_central.off |
| new_wall_03_120v_500c | wall (2) | 2.56e-02 -> 6.01e-02 | [0.0927006258, 0.1061692752, 0.7886506079, 0.0124794911] | new_wall_03_120v_500c_120v_500c_central.off |
| new_wall_04_120v_500c | wall (2) | 9.87e-03 -> 3.11e-02 | [0.1602758505, 0.0778919497, 0.7507703639, 0.0110618359] | new_wall_04_120v_500c_120v_500c_central.off |
| tetbosid | wall (2) | 2.38e-02 -> 1.38e-01 | [0.2567041788, 0.1152573542, 0.499372755, 0.1286657119] | tetbosid_120v_240c_central.off |
| petbosid | wall (2) | 1.32e-02 -> 1.44e-01 | [0.4441959228, 0.1377815878, 0.3017366091, 0.1162858803] | petbosid_120v_230c_central.off |
| gigwebosid | wall (2) | 1.73e-02 -> 1.16e-01 | [0.5439352879, 0.1752013166, 0.1752013166, 0.1056620789] | gigwebosid_120v_320c_central.off |
| new_wall_05_120v_380c | wall (2) | 3.76e-03 -> 6.64e-02 | [0.1052072536, 0.306870764, 0.5266153142, 0.0613066681] | new_wall_05_120v_380c_120v_380c_central.off |
| new_wall_06_120v_380c | wall (2) | 2.47e-02 -> 9.94e-02 | [0.5674139577, 0.0602357644, 0.1642076883, 0.2081425895] | new_wall_06_120v_380c_120v_380c_central.off |
| new_wall_07_120v_380c | wall (2) | 1.68e-02 -> 4.30e-02 | [0.8020014043, 0.0236852924, 0.0735990083, 0.1007142951] | new_wall_07_120v_380c_120v_380c_central.off |
| new_wall_08_120v_440c | wall (2) | 9.04e-03 -> 5.51e-02 | [0.6836340096, 0.0176562184, 0.1606612136, 0.1380485585] | new_wall_08_120v_440c_120v_440c_central.off |
| new_wall_09_120v_440c | wall (2) | 6.17e-03 -> 3.95e-02 | [0.7674073538, 0.0238126824, 0.0641581538, 0.14462181] | new_wall_09_120v_440c_120v_440c_central.off |
| new_wall_10_120v_500c | wall (2) | 1.94e-02 -> 5.61e-02 | [0.7827296914, 0.013233873, 0.0950059866, 0.1090304489] | new_wall_10_120v_500c_120v_500c_central.off |
| new_wall_11_120v_500c | wall (2) | 3.01e-02 -> 5.97e-02 | [0.7138431555, 0.0137977176, 0.1711029184, 0.1012562086] | new_wall_11_120v_500c_120v_500c_central.off |
| new_region_10_120v_680c | region (3) | 7.40e-04 -> 1.58e-03 | [0.0826115348, 0.2879094727, 0.609528041, 0.0199509515] | new_region_10_120v_680c_120v_680c_central.off |
| new_region_11_120v_380c | region (3) | 5.93e-04 -> 5.93e-04 | [0.6001606998, 0.0335291665, 0.299012001, 0.0672981327] | new_region_11_120v_380c_120v_380c_central.off |
| new_wall_12_120v_440c | wall (2) | 4.03e-03 -> 3.03e-02 | [0.1587594812, 0.0887713968, 0.7403118656, 0.0121572564] | new_wall_12_120v_440c_120v_440c_central.off |
| snad | line (1) | 4.43e-02 -> 3.89e-01 | [0.2126938689, 0.2873061311, 0.2873061311, 0.2126938689] | snad_60v_90c_central.off |

## iodecaic

| shape | kind | depth (seed -> centre) | central beta | file |
|---|---|---|---|---|
| sosbid | region (3) | 1.75e-02 -> 5.37e-02 | [0.1617027304, 0.085709939, 0.66682591, 0.0857614206] | sosbid_120v_380c_central.off |
| new_region_01_120v_500c | region (3) | 5.94e-05 -> 1.33e-02 | [0.0772468115, 0.2101987257, 0.6968207257, 0.015733737] | new_region_01_120v_500c_120v_500c_central.off |
| gosbid | region (3) | 6.95e-03 -> 5.83e-02 | [0.3759063591, 0.1024244097, 0.4177712937, 0.1038979375] | gosbid_120v_360c_central.off |
| new_wall_01_120v_260c | wall (2) | 4.35e-02 -> 5.56e-02 | [0.076460401, 0.1752085972, 0.7212837122, 0.0270472895] | new_wall_01_120v_260c_120v_260c_central.off |
| tetosbid | wall (2) | 3.29e-02 -> 1.67e-01 | [0.2645410471, 0.1395430384, 0.427290782, 0.1686251325] | tetosbid_120v_180c_central.off |
| petosbid | wall (2) | 3.15e-02 -> 1.60e-01 | [0.4245947749, 0.1493074719, 0.2971119656, 0.1289857876] | petosbid_120v_170c_central.off |
| gippid | line (1) | 4.26e-02 -> 2.21e-01 | [0.1946487702, 0.3053512298, 0.3053512298, 0.1946487702] | gippid_120v_30c_central.off |

## contic

| shape | kind | depth (seed -> centre) | central beta | file |
|---|---|---|---|---|
| sabotid-like | region (3) | 1.89e-02 -> 1.50e-01 | [0.368439366, 0.3267570087, 0.14781563, 0.1569879953] | sabotid-like_2304v_1248c_central.off |
| gabotid-like | region (3) | 1.47e-02 -> 7.13e-02 | [0.365357948, 0.1105958795, 0.4222048642, 0.1018413083] | gabotid-like_2304v_1056c_central.off |
| gabipard-like | wall (2) | 1.03e-02 -> 1.12e-01 | [0.7051287011, 0.0, 0.2038121933, 0.0910591056] | gabipard-like_1152v_1248c_central.off |
| mabipard-like | wall (2) | 8.82e-02 -> 1.10e-01 | [0.4694413496, 0.0, 0.3254218835, 0.2051367668] | mabipard-like_1152v_1056c_central.off |
| sabipard-like | wall (2) | 3.28e-03 -> 8.61e-02 | [0.1920491377, 0.0, 0.7376559991, 0.0702948632] | sabipard-like_1152v_1248c_central.off |
| mobcated-like | wall (2) | 5.31e-02 -> 1.41e-01 | [0.325094545, 0.1215153254, 0.5533901296, 0.0] | mobcated-like_1152v_864c_central.off |
| sobcated-like | wall (2) | 7.17e-02 -> 1.53e-01 | [0.4084962607, 0.4004277593, 0.19107598, 0.0] | sobcated-like_1152v_1248c_central.off |
| gobcated-like | wall (2) | 5.18e-03 -> 1.05e-01 | [0.1265500283, 0.0943925795, 0.7790573922, 0.0] | gobcated-like_1152v_1056c_central.off |
| petbotid-like | wall (2) | 1.34e-01 -> 1.63e-01 | [0.4237555815, 0.1493409144, 0.2955339062, 0.131369598] | petbotid-like_2304v_384c_central.off |
| tetbotid-like | wall (2) | 1.78e-01 -> 1.81e-01 | [0.2340736493, 0.1711737117, 0.439174646, 0.155577993] | tetbotid-like_2304v_528c_central.off |
| biped-like | line (1) | 2.34e-02 -> 2.05e-01 | [0.6449489743, 0.0, 0.0, 0.3550510257] | biped-like_288v_528c_central.off |
| bited-like | line (1) | 1.18e-01 -> 4.17e-01 | [0.3595388412, 0.6404611588, 0.0, 0.0] | bited-like_384v_1056c_central.off |
| mabred-like | line (1) | 1.09e-01 -> 1.33e-01 | [0.75, 0.0, 0.25, 0.0] | mabred-like_576v_1248c_central.off |
| sabred-like | line (1) | 9.75e-02 -> 5.50e-01 | [0.4661873523, 0.0, 0.5338126477, 0.0] | sabred-like_576v_864c_central.off |
| gabred-like | line (1) | 9.18e-02 -> 1.12e-01 | [0.1335409488, 0.0, 0.8664590512, 0.0] | gabred-like_576v_1056c_central.off |
| bimted-like | line (1) | 2.65e-01 -> 3.45e-01 | [0.0, 0.2892324173, 0.7107675827, 0.0] | bimted-like_576v_192c_central.off |
| tispid-like | line (1) | 9.92e-02 -> 1.48e-01 | [0.6270083225, 0.0, 0.2540166451, 0.1189750324] | tispid-like_1152v_384c_central.off |
| tabcated-like | line (1) | 4.70e-02 -> 4.25e-01 | [0.3333333333, 0.25, 0.4166666667, 0.0] | tabcated-like_1152v_384c_central.off |
| gippid-like | line (1) | 6.05e-02 -> 4.15e-01 | [0.2332813967, 0.2667186033, 0.2667186033, 0.2332813967] | gippid-like_1152v_240c_central.off |
| tabipard-like | line (1) | 6.91e-02 -> 4.55e-01 | [0.3560330591, 0.0, 0.4072000007, 0.2367669402] | tabipard-like_1152v_528c_central.off |
| tadeca-like | line (1) | 1.10e-01 -> 1.68e-01 | [0.1644286286, 0.1370884588, 0.6984829126, 0.0] | tadeca-like_1152v_336c_central.off |
| bideca-like | point (0) | - | [1.0, 0.0, 0.0, 0.0] | bideca-like_48v_288c_central.off |
| spid-like | point (0) | - | [0.5, 0.0, 0.0, 0.5] | spid-like_144v_240c_central.off |
| apid-like | point (0) | - | [0.0, 0.0, 1.0, 0.0] | apid-like_192v_192c_central.off |
| deca-like | point (0) | - | [0.0, 0.5, 0.5, 0.0] | deca-like_288v_48c_central.off |
| respid-like | point (0) | - | [0.6666666667, 0.0, 0.3333333333, 0.0] | respid-like_576v_384c_central.off |
| redeca-like | point (0) | - | [0.2265409197, 0.0, 0.7734590803, 0.0] | redeca-like_576v_336c_central.off |

## kicontic

| shape | kind | depth (seed -> centre) | central beta | file |
|---|---|---|---|---|
| region_01_1152v_5280c | region (3) | 2.01e-02 -> 4.50e-02 | [0.1766456617, 0.0627518781, 0.6505374365, 0.1100650237] | region_01_1152v_5280c_1152v_5280c_central.off |
| region_02_1152v_5280c | region (3) | 3.10e-03 -> 7.41e-03 | [0.0509668881, 0.1860663257, 0.7298080439, 0.0331587423] | region_02_1152v_5280c_1152v_5280c_central.off |
| region_03_1152v_5280c | region (3) | 1.12e-03 -> 1.56e-02 | [0.0975398066, 0.2547457945, 0.6185042205, 0.0292101784] | region_03_1152v_5280c_1152v_5280c_central.off |
| region_04_1152v_3936c | region (3) | 2.32e-03 -> 7.06e-02 | [0.3530988757, 0.0884047472, 0.4313573476, 0.1271390295] | region_04_1152v_3936c_1152v_3936c_central.off |
| region_05_1152v_4128c | region (3) | 1.28e-02 -> 1.11e-01 | [0.3575419749, 0.3583855087, 0.1603932956, 0.1236792207] | region_05_1152v_4128c_1152v_4128c_central.off |
| region_06_1152v_5280c | region (3) | 3.28e-05 -> 6.00e-02 | [0.5753280816, 0.1312829461, 0.2246662482, 0.0687227241] | region_06_1152v_5280c_1152v_5280c_central.off |
| region_07_1152v_5280c | region (3) | 5.18e-03 -> 2.36e-02 | [0.5742752956, 0.0424480853, 0.145017668, 0.2382589511] | region_07_1152v_5280c_1152v_5280c_central.off |
| region_08_1152v_5280c | region (3) | 3.26e-05 -> 3.26e-05 | [0.6242398655, 0.0315738277, 0.0316202075, 0.3125660994] | region_08_1152v_5280c_1152v_5280c_central.off |
| wall_01_1152v_2688c | wall (2) | 3.12e-02 -> 1.13e-01 | [0.1692783456, 0.1150338046, 0.5464095042, 0.1692783456] | wall_01_1152v_2688c_1152v_2688c_central.off |
| wall_02_1152v_3552c | wall (2) | 2.56e-02 -> 3.65e-02 | [0.0698676598, 0.0872940354, 0.816490734, 0.0263475709] | wall_02_1152v_3552c_1152v_3552c_central.off |
| wall_03_1152v_3552c | wall (2) | 1.20e-02 -> 6.11e-02 | [0.0931773331, 0.3225890251, 0.527221589, 0.0570120528] | wall_03_1152v_3552c_1152v_3552c_central.off |
| wall_04_1152v_2256c | wall (2) | 4.58e-02 -> 1.17e-01 | [0.2409025314, 0.0982517414, 0.5531921592, 0.1076535681] | wall_04_1152v_2256c_1152v_2256c_central.off |
| wall_05_1152v_2112c | wall (2) | 1.65e-02 -> 1.45e-01 | [0.4688806349, 0.1191399858, 0.2951812272, 0.1167981521] | wall_05_1152v_2112c_1152v_2112c_central.off |
| wall_06_1152v_2976c | wall (2) | 2.17e-02 -> 1.17e-01 | [0.5458252546, 0.176304487, 0.176304487, 0.1015657714] | wall_06_1152v_2976c_1152v_2976c_central.off |
| wall_07_1152v_3552c | wall (2) | 3.33e-02 -> 1.08e-01 | [0.5454671542, 0.0658868963, 0.1516347948, 0.2370111547] | wall_07_1152v_3552c_1152v_3552c_central.off |
| wall_08_1152v_3552c | wall (2) | 1.06e-02 -> 4.10e-02 | [0.812377731, 0.0205377619, 0.0685697514, 0.0985147556] | wall_08_1152v_3552c_1152v_3552c_central.off |
| region_09_1152v_6432c | region (3) | 3.11e-03 -> 1.46e-02 | [0.0713917134, 0.2378963655, 0.6741230456, 0.0165888754] | region_09_1152v_6432c_1152v_6432c_central.off |
| region_10_1152v_6432c | region (3) | 2.34e-04 -> 1.29e-03 | [0.0760813769, 0.2854255557, 0.6206603197, 0.0178327478] | region_10_1152v_6432c_1152v_6432c_central.off |
| region_11_1152v_6432c | region (3) | 2.10e-03 -> 6.48e-03 | [0.0527516051, 0.1336848361, 0.7957683493, 0.0177952094] | region_11_1152v_6432c_1152v_6432c_central.off |
| region_12_1152v_6432c | region (3) | 2.81e-03 -> 1.45e-02 | [0.6740015596, 0.0140100202, 0.0694072684, 0.2425811518] | region_12_1152v_6432c_1152v_6432c_central.off |
| region_13_1152v_6432c | region (3) | 2.38e-04 -> 1.82e-03 | [0.5962704192, 0.0193289797, 0.0875293642, 0.2968712369] | region_13_1152v_6432c_1152v_6432c_central.off |
| region_14_1152v_6432c | region (3) | 3.17e-03 -> 5.02e-03 | [0.7898838203, 0.0152270336, 0.0453985063, 0.1494906397] | region_14_1152v_6432c_1152v_6432c_central.off |
| snad-like | line (1) | 8.33e-02 -> 3.25e-01 | [0.1625133412, 0.3374866588, 0.3374866588, 0.1625133412] | snad-like_576v_816c_central.off |

## iocontic

| shape | kind | depth (seed -> centre) | central beta | file |
|---|---|---|---|---|
| region_01_1152v_3552c | region (3) | 2.18e-02 -> 4.92e-02 | [0.1533615375, 0.0807780998, 0.682095976, 0.0837643867] | region_01_1152v_3552c_1152v_3552c_central.off |
| region_02_1152v_3360c | region (3) | 2.30e-03 -> 6.33e-02 | [0.3768081563, 0.1099613261, 0.4074553774, 0.1057751401] | region_02_1152v_3360c_1152v_3360c_central.off |
| wall_01_1152v_1680c | wall (2) | 3.15e-02 -> 2.04e-01 | [0.2296672231, 0.1722667577, 0.4496393466, 0.1484266727] | wall_01_1152v_1680c_1152v_1680c_central.off |
| wall_02_1152v_1536c | wall (2) | 9.85e-02 -> 1.74e-01 | [0.4051174156, 0.1609995946, 0.2932336149, 0.1406493749] | wall_02_1152v_1536c_1152v_1536c_central.off |
| region_03_1152v_4704c | region (3) | 1.68e-03 -> 1.21e-02 | [0.0707667971, 0.2043658098, 0.7104459473, 0.0144214457] | region_03_1152v_4704c_1152v_4704c_central.off |
| gippid-like | line (1) | 4.37e-02 -> 5.00e-01 | [0.25, 0.25, 0.25, 0.25] | gippid-like_1152v_240c_central.off |


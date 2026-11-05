# Central seeds and .off files

The central seed of a shape is the point of its locus **deepest inside it**: as far as possible from every place the polytope changes. Where the optimized seed (optimized_off/) makes the edges as even as possible and often sits on or near the shape's edge, the central seed keeps every cell clearly formed.

How (decaic_atlas/central_seeds.py; contic_atlas/central_seeds.py runs it on the contic groups):

* the shape is its locus (the mirror / wall / line equations it lies on) where every sign condition of its polychoron keeps its sign (the forms where a vertex reaches a cell's 3-flat), inside the domain (beta >= 0, b1 >= b4);
* the depth of a seed is its least distance to any of those boundaries: each sign condition is factored and every irreducible factor that changes sign on the chamber is a boundary, its distance f / |grad f| along the locus (exact for a plane);
* a shape on more than one face of the domain (sabipard) is centred on each, the deepest kept;
* the central seed maximises the depth (SLSQP from the map's seed, the optimized seed and their midpoint) and is kept only if it is still the shape: the same arrangement key, and the polychoron rebuilt with the shape's own vertex and cell counts. A point (no freedom) is its own centre.

This replaces an earlier version of this folder that interpolated 35% of the way from the map's seed to the optimized seed: that did not check the result was still the shape (several decaic files were other polytopes, e.g. bimted with 33 cells, biped 79, gabipard 144) and its contic-family files had no hull (0 cells).

Groups: decaic [[3,3,3]] (order 240), kidecaic [[3,3,3]]+ and iodecaic [[3,3,3]+] (120 each); contic [[3,4,3]] (the 24-cell / 48-cell family, order 2304), kicontic [[3,4,3]]+ and iocontic [[3,4,3]+] (1152 each). Seed vectors: R^5 (decaic family, vertices sign * x[perm]) or R^4 (contic family, the standard F4 frame).

| group | files | moved deeper | kept the seed |
|---|---|---|---|
| decaic | 27 | 21 | - |
| kidecaic | 32 | 31 | new_region_11_120v_380c |
| iodecaic | 7 | 7 | - |
| contic | 27 | 21 | - |
| kicontic | 23 | 23 | - |
| iocontic | 6 | 6 | - |
| pennic | 6 | the uniform seeds (= the optimized seeds) | - |

## decaic

| shape | kind | depth (seed -> centre) | central beta | file |
|---|---|---|---|---|
| sabotid | region (3) | 2.19e-02 -> 1.74e-01 | [0.3964014821, 0.3024952049, 0.1505516565, 0.1505516565] | sabotid_240v_140c_central.off |
| gabotid | region (3) | 2.15e-02 -> 1.15e-01 | [0.3788472438, 0.0998046459, 0.4215434645, 0.0998046459] | gabotid_240v_120c_central.off |
| gabipard | wall (2) | 5.10e-02 -> 1.60e-01 | [0.7384168123, 0.0, 0.1307915938, 0.1307915938] | gabipard_120v_140c_central.off |
| mobcated | wall (2) | 1.13e-01 -> 1.76e-01 | [0.3025679385, 0.1436182789, 0.5538137826, 0.0] | mobcated_120v_100c_central.off |
| sobcated | wall (2) | 7.08e-02 -> 2.85e-01 | [0.2324081208, 0.5351837585, 0.2324081208, 0.0] | sobcated_120v_140c_central.off |
| mabipard | wall (2) | 1.54e-01 -> 1.68e-01 | [0.5, 0.0, 0.3628540574, 0.1371459426] | mabipard_120v_120c_central.off |
| sabipard | wall (2) | 2.04e-01 -> 2.59e-01 | [0.5773502692, 0.2113248654, 0.0, 0.2113248654] | sabipard_120v_140c_central.off |
| gobcated | wall (2) | 1.02e-01 -> 1.29e-01 | [0.1056624327, 0.1056624327, 0.7886751346, 0.0] | gobcated_120v_120c_central.off |
| petbotid | wall (2) | 1.76e-01 -> 2.13e-01 | [0.4413214977, 0.1260608735, 0.2608256159, 0.1717920129] | petbotid_240v_50c_central.off |
| tetbotid | wall (2) | 1.86e-01 -> 2.19e-01 | [0.2183017126, 0.1835739068, 0.4779470487, 0.1201773319] | tetbotid_240v_60c_central.off |
| bited | line (1) | 2.36e-01 -> 7.07e-01 | [0.5, 0.5, 0.0, 0.0] | bited_40v_120c_central.off |
| biped | line (1) | 1.01e-01 -> 3.54e-01 | [0.75, 0.0, 0.0, 0.25] | biped_40v_60c_central.off |
| bimted | line (1) | 3.54e-01 -> 3.54e-01 | [0.0, 0.25, 0.75, 0.0] | bimted_60v_30c_central.off |
| gabred | line (1) | 1.77e-01 -> 1.77e-01 | [0.125, 0.0, 0.875, 0.0] | gabred_60v_120c_central.off |
| mabred | line (1) | 1.47e-01 -> 2.36e-01 | [0.8333333333, 0.0, 0.1666666667, 0.0] | mabred_60v_140c_central.off |
| sabred | line (1) | 2.95e-01 -> 2.95e-01 | [0.4583333333, 0.0, 0.5416666667, 0.0] | sabred_60v_100c_central.off |
| tabipard | line (1) | 1.34e-01 -> 4.68e-01 | [0.375, 0.0, 0.375, 0.25] | tabipard_120v_60c_central.off |
| gippid | line (1) | 8.33e-02 -> 5.00e-01 | [0.25, 0.25, 0.25, 0.25] | gippid_120v_30c_central.off |
| tadeca | line (1) | 3.06e-01 -> 3.06e-01 | [0.125, 0.25, 0.625, 0.0] | tadeca_120v_40c_central.off |
| tabcated | line (1) | 9.81e-02 -> 4.25e-01 | [0.3333333333, 0.25, 0.4166666667, 0.0] | tabcated_120v_50c_central.off |
| tispid | line (1) | 3.12e-01 -> 3.12e-01 | [0.5833333333, 0.0, 0.1666666667, 0.25] | tispid_120v_50c_central.off |
| bideca | point (0) | - | [1.0, 0.0, 0.0, 0.0] | bideca_10v_30c_central.off |
| spid | point (0) | - | [0.5, 0.0, 0.0, 0.5] | spid_20v_30c_central.off |
| apid | point (0) | - | [0.0, 0.0, 1.0, 0.0] | apid_20v_30c_central.off |
| deca | point (0) | - | [0.0, 0.5, 0.5, 0.0] | deca_30v_10c_central.off |
| redeca | point (0) | - | [0.25, 0.0, 0.75, 0.0] | redeca_60v_40c_central.off |
| respid | point (0) | - | [0.6666666667, 0.0, 0.3333333333, 0.0] | respid_60v_50c_central.off |

## kidecaic

| shape | kind | depth (seed -> centre) | central beta | file |
|---|---|---|---|---|
| mebosid | region (3) | 1.45e-02 -> 7.00e-02 | [0.1839556293, 0.0606031559, 0.6704501247, 0.0849910901] | mebosid_120v_560c_central.off |
| new_region_01_120v_560c | region (3) | 2.64e-03 -> 1.09e-02 | [0.0537095626, 0.1807189912, 0.7273424571, 0.0382289891] | new_region_01_120v_560c_120v_560c_central.off |
| new_region_02_120v_680c | region (3) | 1.01e-04 -> 8.53e-03 | [0.053496138, 0.1248905708, 0.8053680498, 0.0162452413] | new_region_02_120v_680c_120v_680c_central.off |
| new_region_03_120v_680c | region (3) | 1.55e-03 -> 1.69e-02 | [0.0803291284, 0.2605616962, 0.6444882493, 0.014620926] | new_region_03_120v_680c_120v_680c_central.off |
| new_region_04_120v_560c | region (3) | 1.94e-04 -> 2.07e-02 | [0.1206128867, 0.2365588858, 0.6187477532, 0.0240804743] | new_region_04_120v_560c_120v_560c_central.off |
| apbosid | region (3) | 8.01e-03 -> 1.15e-01 | [0.3788472438, 0.0998046459, 0.4215434645, 0.0998046459] | apbosid_120v_420c_central.off |
| sabosid | region (3) | 1.75e-02 -> 1.59e-01 | [0.3623724357, 0.3623724357, 0.1376275643, 0.1376275643] | sabosid_120v_440c_central.off |
| gabosid | region (3) | 2.62e-05 -> 8.36e-02 | [0.5730417504, 0.1181952145, 0.2363861587, 0.0723768764] | gabosid_120v_560c_central.off |
| new_region_05_120v_560c | region (3) | 2.23e-03 -> 2.99e-02 | [0.5794877903, 0.0300450722, 0.1544180745, 0.236049063] | new_region_05_120v_560c_120v_560c_central.off |
| new_region_06_120v_560c | region (3) | 2.62e-05 -> 8.66e-03 | [0.7245595477, 0.032330911, 0.0445830831, 0.1985264582] | new_region_06_120v_560c_120v_560c_central.off |
| new_region_07_120v_680c | region (3) | 3.07e-04 -> 1.69e-02 | [0.6453426482, 0.014610213, 0.080709578, 0.2593375608] | new_region_07_120v_680c_120v_680c_central.off |
| new_region_08_120v_680c | region (3) | 1.02e-03 -> 6.84e-03 | [0.8022258474, 0.0144107536, 0.0471649995, 0.1361983995] | new_region_08_120v_680c_120v_680c_central.off |
| new_region_09_120v_680c | region (3) | 1.27e-03 -> 2.13e-03 | [0.5897673339, 0.0208793887, 0.0916741398, 0.2976791376] | new_region_09_120v_680c_120v_680c_central.off |
| sigwebosid | wall (2) | 2.50e-02 -> 1.17e-01 | [0.1763044955, 0.1015657714, 0.5458252376, 0.1763044955] | sigwebosid_120v_290c_central.off |
| new_wall_01_120v_380c | wall (2) | 2.28e-02 -> 3.99e-02 | [0.0731566479, 0.0934075343, 0.8041051938, 0.0293306241] | new_wall_01_120v_380c_120v_380c_central.off |
| new_wall_02_120v_440c | wall (2) | 3.90e-02 -> 4.24e-02 | [0.0556157482, 0.151340407, 0.7664802876, 0.0265635571] | new_wall_02_120v_440c_120v_440c_central.off |
| new_wall_03_120v_500c | wall (2) | 2.10e-02 -> 7.37e-02 | [0.0605500722, 0.1175653428, 0.8131300319, 0.008754553] | new_wall_03_120v_500c_120v_500c_central.off |
| new_wall_04_120v_500c | wall (2) | 7.89e-03 -> 4.20e-02 | [0.1844744096, 0.0616072067, 0.7477124625, 0.0062059212] | new_wall_04_120v_500c_120v_500c_central.off |
| tetbosid | wall (2) | 4.64e-02 -> 1.89e-01 | [0.2377093296, 0.1584863736, 0.4698992639, 0.1339050329] | tetbosid_120v_240c_central.off |
| petbosid | wall (2) | 1.39e-02 -> 2.08e-01 | [0.441545145, 0.1268635237, 0.263680861, 0.1679104704] | petbosid_120v_230c_central.off |
| gigwebosid | wall (2) | 1.75e-02 -> 1.17e-01 | [0.545825252, 0.1763044883, 0.1763044883, 0.1015657714] | gigwebosid_120v_320c_central.off |
| new_wall_05_120v_380c | wall (2) | 3.77e-03 -> 7.23e-02 | [0.1116925826, 0.3015924083, 0.5222089641, 0.0645060451] | new_wall_05_120v_380c_120v_380c_central.off |
| new_wall_06_120v_380c | wall (2) | 4.74e-02 -> 1.09e-01 | [0.6217641329, 0.0435485697, 0.1977896344, 0.136897663] | new_wall_06_120v_380c_120v_380c_central.off |
| new_wall_07_120v_380c | wall (2) | 1.68e-02 -> 4.46e-02 | [0.8007952139, 0.0247983916, 0.0694284635, 0.104977931] | new_wall_07_120v_380c_120v_380c_central.off |
| new_wall_08_120v_440c | wall (2) | 8.26e-03 -> 5.71e-02 | [0.6855007903, 0.0158636062, 0.1706293761, 0.1280062274] | new_wall_08_120v_440c_120v_440c_central.off |
| new_wall_09_120v_440c | wall (2) | 5.92e-03 -> 4.15e-02 | [0.7631043382, 0.0235506038, 0.051678316, 0.161666742] | new_wall_09_120v_440c_120v_440c_central.off |
| new_wall_10_120v_500c | wall (2) | 1.62e-02 -> 7.37e-02 | [0.8131300333, 0.0087545529, 0.0605500723, 0.1175653415] | new_wall_10_120v_500c_120v_500c_central.off |
| new_wall_11_120v_500c | wall (2) | 2.23e-02 -> 7.65e-02 | [0.690494668, 0.0097495172, 0.2024658362, 0.0972899787] | new_wall_11_120v_500c_120v_500c_central.off |
| new_region_10_120v_680c | region (3) | 7.40e-04 -> 1.62e-03 | [0.0833356274, 0.2866325792, 0.6102536638, 0.0197781296] | new_region_10_120v_680c_120v_680c_central.off |
| new_region_11_120v_380c | region (3) | 6.00e-04 -> 6.00e-04 | [0.6001606998, 0.0335291665, 0.299012001, 0.0672981327] | new_region_11_120v_380c_120v_380c_central.off |
| new_wall_12_120v_440c | wall (2) | 3.74e-03 -> 2.95e-02 | [0.1688669709, 0.071006978, 0.7506255834, 0.0095004677] | new_wall_12_120v_440c_120v_440c_central.off |
| snad | line (1) | 8.33e-02 -> 5.00e-01 | [0.25, 0.25, 0.25, 0.25] | snad_60v_90c_central.off |

## iodecaic

| shape | kind | depth (seed -> centre) | central beta | file |
|---|---|---|---|---|
| sosbid | region (3) | 1.75e-02 -> 7.27e-02 | [0.1715418958, 0.062983554, 0.6967843673, 0.0686901828] | sosbid_120v_380c_central.off |
| new_region_01_120v_500c | region (3) | 5.94e-05 -> 1.59e-02 | [0.075033489, 0.1996357526, 0.7115793482, 0.0137514102] | new_region_01_120v_500c_120v_500c_central.off |
| gosbid | region (3) | 8.01e-03 -> 1.15e-01 | [0.3788472438, 0.0998046459, 0.4215434645, 0.0998046459] | gosbid_120v_360c_central.off |
| new_wall_01_120v_260c | wall (2) | 4.47e-02 -> 5.86e-02 | [0.0769299502, 0.1710209344, 0.72507024, 0.0269788755] | new_wall_01_120v_260c_120v_260c_central.off |
| tetosbid | wall (2) | 3.63e-02 -> 2.11e-01 | [0.2289844762, 0.176612286, 0.4598219994, 0.1345812384] | tetosbid_120v_180c_central.off |
| petosbid | wall (2) | 3.15e-02 -> 2.10e-01 | [0.4344297832, 0.1319003391, 0.2645605837, 0.169109294] | petosbid_120v_170c_central.off |
| gippid | line (1) | 8.33e-02 -> 5.00e-01 | [0.25, 0.25, 0.25, 0.25] | gippid_120v_30c_central.off |

## contic

| shape | kind | depth (seed -> centre) | central beta | file |
|---|---|---|---|---|
| sabotid-like | region (3) | 2.28e-02 -> 1.74e-01 | [0.3964014821, 0.3024952049, 0.1505516565, 0.1505516565] | sabotid-like_2304v_1248c_central.off |
| gabotid-like | region (3) | 1.57e-02 -> 1.18e-01 | [0.3710653383, 0.1025031098, 0.4239284422, 0.1025031098] | gabotid-like_2304v_1056c_central.off |
| gabipard-like | wall (2) | 2.10e-02 -> 1.60e-01 | [0.7384168123, 0.0, 0.1307915938, 0.1307915938] | gabipard-like_1152v_1248c_central.off |
| mabipard-like | wall (2) | 1.54e-01 -> 1.73e-01 | [0.4952833613, 0.0, 0.3636894889, 0.1410271497] | mabipard-like_1152v_1056c_central.off |
| sabipard-like | wall (2) | 2.04e-01 -> 2.59e-01 | [0.5773502692, 0.2113248654, 0.0, 0.2113248654] | sabipard-like_1152v_1248c_central.off |
| mobcated-like | wall (2) | 5.66e-02 -> 1.83e-01 | [0.2883522098, 0.149225655, 0.5624221352, 0.0] | mobcated-like_1152v_864c_central.off |
| sobcated-like | wall (2) | 9.58e-02 -> 2.85e-01 | [0.2324081208, 0.5351837585, 0.2324081208, 0.0] | sobcated-like_1152v_1248c_central.off |
| gobcated-like | wall (2) | 5.10e-03 -> 1.20e-01 | [0.0976310729, 0.0976310729, 0.8047378541, 0.0] | gobcated-like_1152v_1056c_central.off |
| petbotid-like | wall (2) | 1.76e-01 -> 2.13e-01 | [0.4413214977, 0.1260608735, 0.2608256159, 0.1717920129] | petbotid-like_2304v_384c_central.off |
| tetbotid-like | wall (2) | 1.82e-01 -> 2.06e-01 | [0.220422338, 0.1740550244, 0.472518115, 0.1330045227] | tetbotid-like_2304v_528c_central.off |
| biped-like | line (1) | 6.90e-02 -> 3.54e-01 | [0.75, 0.0, 0.0, 0.25] | biped-like_288v_528c_central.off |
| bited-like | line (1) | 2.07e-01 -> 7.07e-01 | [0.5, 0.5, 0.0, 0.0] | bited-like_384v_1056c_central.off |
| mabred-like | line (1) | 1.73e-01 -> 2.36e-01 | [0.8333333333, 0.0, 0.1666666667, 0.0] | mabred-like_576v_1248c_central.off |
| sabred-like | line (1) | 1.01e-01 -> 3.11e-01 | [0.4466037932, 0.0, 0.5533962068, 0.0] | sabred-like_576v_864c_central.off |
| gabred-like | line (1) | 1.60e-01 -> 1.60e-01 | [0.1132704598, 0.0, 0.8867295402, 0.0] | gabred-like_576v_1056c_central.off |
| bimted-like | line (1) | 3.54e-01 -> 3.54e-01 | [0.0, 0.25, 0.75, 0.0] | bimted-like_576v_192c_central.off |
| tispid-like | line (1) | 1.56e-01 -> 2.08e-01 | [0.6111111111, 0.0, 0.2222222222, 0.1666666667] | tispid-like_1152v_384c_central.off |
| tabcated-like | line (1) | 8.70e-02 -> 4.25e-01 | [0.3333333333, 0.25, 0.4166666667, 0.0] | tabcated-like_1152v_384c_central.off |
| gippid-like | line (1) | 1.13e-01 -> 5.00e-01 | [0.25, 0.25, 0.25, 0.25] | gippid-like_1152v_240c_central.off |
| tabipard-like | line (1) | 1.32e-01 -> 4.80e-01 | [0.3632704598, 0.0, 0.3867295402, 0.25] | tabipard-like_1152v_528c_central.off |
| tadeca-like | line (1) | 3.07e-01 -> 3.07e-01 | [0.1132704598, 0.25, 0.6367295402, 0.0] | tadeca-like_1152v_336c_central.off |
| bideca-like | point (0) | - | [1.0, 0.0, 0.0, 0.0] | bideca-like_48v_288c_central.off |
| spid-like | point (0) | - | [0.5, 0.0, 0.0, 0.5] | spid-like_144v_240c_central.off |
| apid-like | point (0) | - | [0.0, 0.0, 1.0, 0.0] | apid-like_192v_192c_central.off |
| deca-like | point (0) | - | [0.0, 0.5, 0.5, 0.0] | deca-like_288v_48c_central.off |
| respid-like | point (0) | - | [0.6666666667, 0.0, 0.3333333333, 0.0] | respid-like_576v_384c_central.off |
| redeca-like | point (0) | - | [0.2265409197, 0.0, 0.7734590803, 0.0] | redeca-like_576v_336c_central.off |

## kicontic

| shape | kind | depth (seed -> centre) | central beta | file |
|---|---|---|---|---|
| region_01_1152v_5280c | region (3) | 2.01e-02 -> 6.52e-02 | [0.1683279799, 0.0564451661, 0.6990734443, 0.0761534096] | region_01_1152v_5280c_1152v_5280c_central.off |
| region_02_1152v_5280c | region (3) | 3.45e-03 -> 1.03e-02 | [0.0499816822, 0.1755395338, 0.739102189, 0.035376595] | region_02_1152v_5280c_1152v_5280c_central.off |
| region_03_1152v_5280c | region (3) | 1.09e-03 -> 1.86e-02 | [0.1085395355, 0.2396071287, 0.6302850808, 0.021568255] | region_03_1152v_5280c_1152v_5280c_central.off |
| region_04_1152v_3936c | region (3) | 2.31e-03 -> 1.18e-01 | [0.3710653383, 0.1025031098, 0.4239284422, 0.1025031098] | region_04_1152v_3936c_1152v_3936c_central.off |
| region_05_1152v_4128c | region (3) | 2.18e-02 -> 1.59e-01 | [0.3623724357, 0.3623724357, 0.1376275643, 0.1376275643] | region_05_1152v_4128c_1152v_4128c_central.off |
| region_06_1152v_5280c | region (3) | 3.28e-05 -> 8.36e-02 | [0.5730417504, 0.1181952145, 0.2363861587, 0.0723768764] | region_06_1152v_5280c_1152v_5280c_central.off |
| region_07_1152v_5280c | region (3) | 3.43e-03 -> 3.02e-02 | [0.5802437453, 0.0294877313, 0.1534249247, 0.2368435987] | region_07_1152v_5280c_1152v_5280c_central.off |
| region_08_1152v_5280c | region (3) | 3.28e-05 -> 7.62e-03 | [0.7332708721, 0.0284454467, 0.0392188215, 0.1990648597] | region_08_1152v_5280c_1152v_5280c_central.off |
| wall_01_1152v_2688c | wall (2) | 3.12e-02 -> 1.17e-01 | [0.176304507, 0.1015657714, 0.5458252147, 0.176304507] | wall_01_1152v_2688c_1152v_2688c_central.off |
| wall_02_1152v_3552c | wall (2) | 2.75e-02 -> 5.49e-02 | [0.0787725383, 0.0653807257, 0.8358528626, 0.0199938734] | wall_02_1152v_3552c_1152v_3552c_central.off |
| wall_03_1152v_3552c | wall (2) | 1.20e-02 -> 6.60e-02 | [0.0992997275, 0.3159910565, 0.5249344673, 0.0597747486] | wall_03_1152v_3552c_1152v_3552c_central.off |
| wall_04_1152v_2256c | wall (2) | 4.58e-02 -> 1.19e-01 | [0.2314720101, 0.1004673207, 0.5758146759, 0.0922459932] | wall_04_1152v_2256c_1152v_2256c_central.off |
| wall_05_1152v_2112c | wall (2) | 1.77e-02 -> 1.69e-01 | [0.4452577179, 0.1319135249, 0.2862560106, 0.1365727466] | wall_05_1152v_2112c_1152v_2112c_central.off |
| wall_06_1152v_2976c | wall (2) | 3.66e-02 -> 1.95e-01 | [0.5380883073, 0.146538002, 0.146538002, 0.1688356888] | wall_06_1152v_2976c_1152v_2976c_central.off |
| wall_07_1152v_3552c | wall (2) | 5.67e-02 -> 1.30e-01 | [0.566491567, 0.0590754528, 0.1474136215, 0.2270193587] | wall_07_1152v_3552c_1152v_3552c_central.off |
| wall_08_1152v_3552c | wall (2) | 1.14e-02 -> 4.30e-02 | [0.8109558331, 0.0216989213, 0.0637081641, 0.1036370815] | wall_08_1152v_3552c_1152v_3552c_central.off |
| region_09_1152v_6432c | region (3) | 1.92e-03 -> 1.52e-02 | [0.0738991224, 0.2583742694, 0.654544292, 0.0131823162] | region_09_1152v_6432c_1152v_6432c_central.off |
| region_10_1152v_6432c | region (3) | 2.34e-04 -> 1.32e-03 | [0.0766684967, 0.2843130544, 0.6213306111, 0.0176878377] | region_10_1152v_6432c_1152v_6432c_central.off |
| region_11_1152v_6432c | region (3) | 2.10e-03 -> 8.08e-03 | [0.0501544119, 0.1211170738, 0.8137543152, 0.0149741991] | region_11_1152v_6432c_1152v_6432c_central.off |
| region_12_1152v_6432c | region (3) | 2.84e-03 -> 1.52e-02 | [0.6564463427, 0.0131371786, 0.0744734977, 0.255942981] | region_12_1152v_6432c_1152v_6432c_central.off |
| region_13_1152v_6432c | region (3) | 2.38e-04 -> 1.90e-03 | [0.5961670905, 0.0190899194, 0.0882425889, 0.2965004012] | region_13_1152v_6432c_1152v_6432c_central.off |
| region_14_1152v_6432c | region (3) | 3.17e-03 -> 6.07e-03 | [0.808425594, 0.0128424457, 0.042575361, 0.1361565993] | region_14_1152v_6432c_1152v_6432c_central.off |
| snad-like | line (1) | 8.33e-02 -> 5.00e-01 | [0.25, 0.25, 0.25, 0.25] | snad-like_576v_816c_central.off |

## iocontic

| shape | kind | depth (seed -> centre) | central beta | file |
|---|---|---|---|---|
| region_01_1152v_3552c | region (3) | 2.18e-02 -> 6.73e-02 | [0.157731637, 0.0582484908, 0.7214076224, 0.0626122498] | region_01_1152v_3552c_1152v_3552c_central.off |
| region_02_1152v_3360c | region (3) | 2.31e-03 -> 1.18e-01 | [0.3710653383, 0.1025031098, 0.4239284422, 0.1025031098] | region_02_1152v_3360c_1152v_3360c_central.off |
| wall_01_1152v_1680c | wall (2) | 3.15e-02 -> 2.09e-01 | [0.2236100721, 0.1761266143, 0.4597140902, 0.1405492235] | wall_01_1152v_1680c_1152v_1680c_central.off |
| wall_02_1152v_1536c | wall (2) | 9.85e-02 -> 2.13e-01 | [0.4413214977, 0.1260608735, 0.2608256159, 0.1717920129] | wall_02_1152v_1536c_1152v_1536c_central.off |
| region_03_1152v_4704c | region (3) | 1.68e-03 -> 1.45e-02 | [0.068783503, 0.1926332822, 0.7260112295, 0.0125719853] | region_03_1152v_4704c_1152v_4704c_central.off |
| gippid-like | line (1) | 4.37e-02 -> 5.00e-01 | [0.25, 0.25, 0.25, 0.25] | gippid-like_1152v_240c_central.off |

## pennic

| shape | central beta | vertices, cells |
|---|---|---|
| pen | [1.0, 0.0, 0.0, 0.0] | [5, 5] |
| rap | [0.0, 1.0, 0.0, 0.0] | [10, 10] |
| tip | [0.5, 0.5, 0.0, 0.0] | [20, 10] |
| srip | [0.5, 0.0, 0.5, 0.0] | [30, 20] |
| grip | [0.333333, 0.333333, 0.333333, 0.0] | [60, 20] |
| prip | [0.333333, 0.333333, 0.0, 0.333333] | [60, 30] |

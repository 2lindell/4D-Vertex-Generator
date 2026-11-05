# Decaic shapes ([[3,3,3]], order 240)

27 shapes found; accepted list: 27; all names and degrees match: **True**.

Seeds are beta = (b1, b2, b3, b4) in the fundamental-weight basis (beta_k = x_k - x_{k+1}); domain beta >= 0, b1 >= b4.

| name | kind (degrees) | vertices | cells | seed beta | example face lies on |
|---|---|---|---|---|---|
| sabotid | region (3) | 240 | 140: 10x24v(6x4+8x6) + 40x12v(6x4+2x6) + 90x8v(6x4) | (17132, 1610, 455, 4827) | (open region) |
| gabotid | region (3) | 240 | 120: 10x24v(6x4+8x6) + 80x12v(6x4+2x6) + 30x8v(6x4) | (3879, 2905, 7347, 1885) | (open region) |
| gabipard | wall (2) | 120 | 140: 10x12v(4x3+4x6) + 20x12v(6x4+2x6) + 30x8v(6x4) + 60x6v(2x3+3x4) + 20x6v(8x3) | (19, 0, 1, 4) | b2 = 0 |
| mobcated | wall (2) | 120 | 100: 10x12v(4x3+4x6) + 20x12v(6x4+2x6) + 40x9v(4x3+3x4+1x6) + 30x4v(4x3) | (2, 1, 3, 0) | b4 = 0 |
| sobcated | wall (2) | 120 | 140: 10x12v(4x3+4x6) + 20x12v(6x4+2x6) + 80x6v(2x3+3x4) + 30x4v(4x3) | (7, 8, 9, 0) | b4 = 0 |
| mabipard | wall (2) | 120 | 120: 10x12v(4x3+4x6) + 40x9v(4x3+3x4+1x6) + 30x8v(6x4) + 20x6v(2x3+3x4) + 20x6v(8x3) | (3, 0, 2, 1) | b2 = 0 |
| sabipard | wall (2) | 120 | 140: 10x12v(8x3+6x4) + 30x8v(6x4) + 80x6v(2x3+3x4) + 20x6v(8x3) | (3, 0, 7, 2) | b2 = 0 |
| gobcated | wall (2) | 120 | 120: 10x24v(6x4+8x6) + 20x12v(6x4+2x6) + 60x6v(2x3+3x4) + 30x4v(4x3) | (1, 2, 9, 0) | b4 = 0 |
| petbotid | wall (2) | 240 | 50: 10x24v(6x4+8x6) + 20x24v(9x4+2x6+3x8) + 20x12v(6x4+2x6) | (7, 3, 5, 3) | b1 + 2*b2 = 2*b3 + b4 |
| tetbotid | wall (2) | 240 | 60: 10x48v(18x4+4x6+4x12) + 20x12v(6x4+2x6) + 30x8v(6x4) | (3, 2, 5, 2) | 3*b1 + b2 = b3 + 3*b4 |
| bited | line (1) | 40 | 120: 20x6v(8x3) + 100x4v(4x3) | (1, 5, 0, 0) | b4 = 0; b3 = 0 |
| biped | line (1) | 40 | 60: 30x8v(6x4) + 20x6v(2x3+3x4) + 10x4v(4x3) | (13, 0, 0, 1) | b3 = 0; b2 = b3; b2 = 0 |
| bimted | line (1) | 60 | 30: 10x12v(4x3+4x6) + 20x12v(6x4+2x6) | (0, 1, 3, 0) | b4 = 0; b1 = b4; b1 = 0 |
| gabred | line (1) | 60 | 120: 10x12v(8x3+6x4) + 20x6v(8x3) + 60x5v(4x3+1x4) + 30x4v(4x3) | (1, 0, 7, 0) | b4 = 0; b2 = 0 |
| mabred | line (1) | 60 | 140: 20x6v(2x3+3x4) + 30x6v(8x3) + 60x5v(4x3+1x4) + 30x4v(4x3) | (37, 0, 11, 0) | b4 = 0; b2 = 0 |
| sabred | line (1) | 60 | 100: 70x6v(8x3) + 30x4v(4x3) | (11, 0, 13, 0) | b4 = 0; b2 = 0 |
| tabipard | line (1) | 120 | 60: 10x24v(4x3+12x4+4x6) + 30x8v(6x4) + 20x6v(2x3+3x4) | (13, 0, 3, 12) | b2 = 0; 3*b1 = b3 + 3*b4; 3*b1 + b2 = b3 + 3*b4 |
| gippid | line (1) | 120 | 30: 10x24v(6x4+8x6) + 20x12v(6x4+2x6) | (1, 11, 11, 1) | b2 = b3; b1 = b4; b1 + 2*b2 = 2*b3 + b4; 3*b1 + b2 = b3 + 3*b4 |
| tadeca | line (1) | 120 | 40: 10x36v(12x3+4x6+4x12) + 30x4v(4x3) | (1, 2, 5, 0) | b4 = 0; 3*b1 + b2 = b3 + 3*b4; 3*b1 + b2 = b3; 3*b1 + b2 + 2*b4 = b3 |
| tabcated | line (1) | 120 | 50: 20x12v(2x3+9x4) + 10x12v(4x3+4x6) + 20x12v(6x4+2x6) | (4, 23, 25, 0) | b4 = 0; b1 + 2*b2 = 2*b3 + b4; b1 + 2*b2 = 2*b3 |
| tispid | line (1) | 120 | 50: 20x18v(6x3+2x6+3x8) + 10x12v(4x3+4x6) + 20x6v(8x3) | (7, 0, 2, 3) | b2 = 0; b1 = 3*b2 + 2*b3 + b4; b1 = 2*b3 + b4; b1 + 2*b2 = 2*b3 + b4 |
| bideca | point (0) | 10 | 30: 30x4v(4x3) | (1, 0, 0, 0) | b4 = 0; b3 = 0; b2 = b3; b2 = 0 |
| spid | point (0) | 20 | 30: 20x6v(2x3+3x4) + 10x4v(4x3) | (1, 0, 0, 1) | b3 = 0; b2 = b3; b2 = 0; b1 = 3*b2 + 7*b3 + b4; b1 = 3*b2 + 2*b3 + b4; b1 = 7*b3 + b4; b1 = 2*b3 + b4; b1 = b4; b1 + 2*b2 = 2*b3 + b4; b1 + 2*b2 = b4; b1 + 2*b2 + 3*b3 = b4; b1 + 7*b2 + 3*b3 = b4; 3*b1 = b3 + 3*b4; 3*b1 + b2 = b3 + 3*b4; 3*b1 + b2 = 3*b4 |
| apid | point (0) | 20 | 30: 20x6v(8x3) + 10x4v(4x3) | (0, 0, 1, 0) | b4 = 0; b2 = 0; b1 = b4; b1 = 0; b1 + 2*b2 = b4; 3*b1 + b2 = 3*b4 |
| deca | point (0) | 30 | 10: 10x12v(4x3+4x6) | (0, 1, 1, 0) | b4 = 0; b2 = b3; b1 = b4; b1 = 0; b1 + 2*b2 = 2*b3 + b4; b1 + 2*b2 = 2*b3; 2*b1 + b3 = b2; 2*b1 + b3 + 3*b4 = b2; 3*b1 + b2 = b3 + 3*b4; 3*b1 + b2 = b3; 3*b1 + b2 + 2*b4 = b3 |
| redeca | point (0) | 60 | 40: 10x18v(16x3+4x6) + 30x4v(4x3) | (1, 0, 3, 0) | b4 = 0; b2 = 0; 3*b1 = b3 + 3*b4; 3*b1 = b3; 3*b1 + 2*b4 = b3; 3*b1 + b2 = b3 + 3*b4; 3*b1 + b2 = b3; 3*b1 + b2 + 2*b4 = b3 |
| respid | point (0) | 60 | 50: 20x9v(8x3+3x4) + 30x6v(8x3) | (2, 0, 1, 0) | b4 = 0; b2 = 0; b1 = 3*b2 + 2*b3 + b4; b1 = 2*b3 + b4; b1 = 2*b3; b1 + 2*b2 = 2*b3 + b4; b1 + 2*b2 = 2*b3 |

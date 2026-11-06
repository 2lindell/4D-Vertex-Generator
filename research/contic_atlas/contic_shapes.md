# Contic shapes ([[3,4,3]], order 2304)

27 shapes; each paired with its decaic counterpart (missing: none).

Seeds: beta = distances to the four mirrors of the F4 chamber (b1, b2 long-root mirrors, b3, b4 short); domain beta >= 0, b1 >= b4. Exact values a + b*sqrt2; the float shown is normalized to sum 1.

| name | decaic counterpart | kind | vertices | cells | seed beta (float) | example face lies on |
|---|---|---|---|---|---|---|
| sabotic | sabotid | region (3) | 2304 | 1248: 48x48v(12x4+8x6+6x8) + 144x16v(8x4+2x8) + 192x12v(6x4+2x6) + 864x8v(6x4) | [0.912713, 0.043169, 0.01971, 0.024408] | (open region) |
| gabotic | gabotid | region (3) | 2304 | 1056: 48x48v(12x4+8x6+6x8) + 144x16v(8x4+2x8) + 576x12v(6x4+2x6) + 288x8v(6x4) | [0.297526, 0.240762, 0.366835, 0.094877] | (open region) |
| gabiparc | gabipard | wall (2) | 1152 | 1248: 48x24v(6x4+8x6) + 192x12v(6x4+2x6) + 288x8v(6x4) + 144x8v(8x3+2x4) + 576x6v(2x3+3x4) | [0.816226, 0.0, 0.017107, 0.166667] | b2 = 0 |
| mabiparc | mabipard | wall (2) | 1152 | 1056: 48x24v(6x4+8x6) + 384x9v(4x3+3x4+1x6) + 288x8v(6x4) + 144x8v(8x3+2x4) + 192x6v(2x3+3x4) | [0.5, 0.0, 0.333333, 0.166667] | b2 = 0 |
| sabiparc | sabipard | wall (2) | 1152 | 1248: 48x24v(8x3+18x4) + 288x8v(6x4) + 144x8v(8x3+2x4) + 768x6v(2x3+3x4) | [0.314081, 0.0, 0.519253, 0.166667] | b2 = 0 |
| mobcotic | mobcated | wall (2) | 1152 | 864: 48x24v(8x3+6x8) + 144x16v(8x4+2x8) + 384x9v(4x3+3x4+1x6) + 288x4v(4x3) | [0.166667, 0.333333, 0.5, 0.0] | b4 = 0 |
| sobcotic | sobcated | wall (2) | 1152 | 1248: 48x24v(8x3+6x8) + 144x16v(8x4+2x8) + 768x6v(2x3+3x4) + 288x4v(4x3) | [0.316226, 0.333333, 0.35044, 0.0] | b4 = 0 |
| gobcotic | gobcated | wall (2) | 1152 | 1056: 48x48v(12x4+8x6+6x8) + 144x16v(8x4+2x8) + 576x6v(2x3+3x4) + 288x4v(4x3) | [0.147414, 0.166667, 0.685919, 0.0] | b4 = 0 |
| pitbotic | petbotid | wall (2) | 2304 | 384: 48x48v(12x4+8x6+6x8) + 192x24v(9x4+2x6+3x8) + 144x16v(8x4+2x8) | [0.388889, 0.166667, 0.277778, 0.166667] | b1 + 2*b2 = 2*b3 + b4 |
| catbotic | tetbotid | wall (2) | 2304 | 528: 48x96v(36x4+8x6+6x16) + 192x12v(6x4+2x6) + 288x8v(6x4) | [0.24218, 0.166667, 0.424486, 0.166667] | b1 + (1-1/2*sqrt2)*b2 = (1-1/2*sqrt2)*b3 + b4 |
| bipec | biped | line (1) | 288 | 528: 288x8v(6x4) + 192x6v(2x3+3x4) + 48x6v(8x3) | [0.951184, 0.0, 0.0, 0.048816] | b2 = 0; b3 = 0; b2 = b3 |
| bitec | bited | line (1) | 384 | 1056: 48x8v(6x4) + 144x8v(8x3+2x4) + 864x4v(4x3) | [0.146447, 0.853553, 0.0, 0.0] | b3 = 0; b4 = 0 |
| sabric | mabred | line (1) | 576 | 1248: 48x12v(8x3+6x4) + 144x8v(8x3+2x4) + 192x6v(2x3+3x4) + 576x5v(4x3+1x4) + 288x4v(4x3) | [0.78886, 0.0, 0.21114, 0.0] | b2 = 0; b4 = 0 |
| mabric | sabred | line (1) | 576 | 864: 48x12v(8x3+6x4) + 144x8v(8x3+2x4) + 384x6v(8x3) + 288x4v(4x3) | [0.297969, 0.0, 0.702031, 0.0] | b2 = 0; b4 = 0 |
| gabric | gabred | line (1) | 576 | 1056: 48x24v(8x3+18x4) + 144x8v(8x3+2x4) + 576x5v(4x3+1x4) + 288x4v(4x3) | [0.11327, 0.0, 0.88673, 0.0] | b2 = 0; b4 = 0 |
| bimtec | bimted | line (1) | 576 | 192: 48x24v(8x3+6x8) + 144x16v(8x4+2x8) | [0.0, 0.25, 0.75, 0.0] | b1 = 0; b4 = 0; b1 = b4 |
| tispic | tispid | line (1) | 1152 | 384: 48x24v(6x4+8x6) + 192x18v(6x3+2x6+3x8) + 144x8v(8x3+2x4) | [0.583333, 0.0, 0.166667, 0.25] | b2 = 0; b1 = (1+2*sqrt2)*b2 + 2*b3 + b4; b1 = 2*b3 + b4; b1 + 2*b2 = 2*b3 + b4 |
| tabcatic | tabcated | line (1) | 1152 | 384: 48x24v(8x3+6x8) + 144x16v(8x4+2x8) + 192x12v(2x3+9x4) | [0.068227, 0.448829, 0.482943, 0.0] | b4 = 0; b1 + 2*b2 = 2*b3 + b4; b1 + 2*b2 = 2*b3 |
| gippic | gippid | line (1) | 1152 | 240: 48x48v(12x4+8x6+6x8) + 192x12v(6x4+2x6) | [0.056635, 0.443365, 0.443365, 0.056635] | b1 = b4; b2 = b3; b1 + (-1+1*sqrt2)*b3 = (-1+1*sqrt2)*b2 + b4; b1 + (1-1/2*sqrt2)*b2 = (1-1/2*sqrt2)*b3 + b4; b1 + 2*b2 = 2*b3 + b4 |
| tabiparc | tabipard | line (1) | 1152 | 528: 48x48v(8x3+24x4+6x8) + 288x8v(6x4) + 192x6v(2x3+3x4) | [0.462445, 0.0, 0.106222, 0.431333] | b2 = 0; b1 = (1-1/2*sqrt2)*b3 + b4; b1 + (1-1/2*sqrt2)*b2 = (1-1/2*sqrt2)*b3 + b4 |
| ticont | tadeca | line (1) | 1152 | 336: 48x72v(24x3+8x6+6x16) + 288x4v(4x3) | [0.11327, 0.25, 0.63673, 0.0] | b4 = 0; b1 + (1-1/2*sqrt2)*b2 = (1-1/2*sqrt2)*b3 + b4; b1 + (1-1/2*sqrt2)*b2 = (1-1/2*sqrt2)*b3; b1 + (1-1/2*sqrt2)*b2 + 1/2*sqrt2*b4 = (1-1/2*sqrt2)*b3 |
| bicont | bideca | point (0) | 48 | 288: 288x4v(4x3) | [1.0, 0.0, 0.0, 0.0] | b2 = 0; b3 = 0; b4 = 0; b2 = b3 |
| spic | spid | point (0) | 144 | 240: 192x6v(2x3+3x4) + 48x6v(8x3) | [0.5, 0.0, 0.0, 0.5] | b2 = 0; b3 = 0; b1 = b4; b2 = b3; b1 = (1+2*sqrt2)*b2 + (6+3*sqrt2)*b3 + b4; b1 = (1+2*sqrt2)*b2 + 2*b3 + b4; b1 + (-1+1*sqrt2)*b3 = (-1+1*sqrt2)*b2 + b4; b1 = 2*b3 + b4; b1 = (3/2-1/2*sqrt2)*b3 + b4; b1 = (1-1/2*sqrt2)*b3 + b4; b1 + (-1+1*sqrt2)*b3 = b4; b1 + (1-1/2*sqrt2)*b2 = (1-1/2*sqrt2)*b3 + b4; b1 + (1-1/2*sqrt2)*b2 = b4; b1 + 2*b2 = 2*b3 + b4; b1 + 2*b2 = b4; b1 + 2*b2 + (1+2*sqrt2)*b3 = b4; b1 + (6+3*sqrt2)*b2 + (1+2*sqrt2)*b3 = b4 |
| bamic | apid | point (0) | 192 | 192: 48x8v(6x4) + 144x8v(8x3+2x4) | [0.0, 0.0, 1.0, 0.0] | b1 = 0; b2 = 0; b4 = 0; b1 = b4; b1 + (1-1/2*sqrt2)*b2 = b4; b1 + 2*b2 = b4 |
| cont | deca | point (0) | 288 | 48: 48x24v(8x3+6x8) | [0.0, 0.5, 0.5, 0.0] | b1 = 0; b4 = 0; b1 = b4; b2 = b3; b1 + (-1+1*sqrt2)*b3 = (-1+1*sqrt2)*b2 + b4; b1 + (-1+1*sqrt2)*b3 = (-1+1*sqrt2)*b2; b1 + (-1+1*sqrt2)*b3 + 1*sqrt2*b4 = (-1+1*sqrt2)*b2; b1 + (1-1/2*sqrt2)*b2 = (1-1/2*sqrt2)*b3 + b4; b1 + (1-1/2*sqrt2)*b2 = (1-1/2*sqrt2)*b3; b1 + (1-1/2*sqrt2)*b2 + 1/2*sqrt2*b4 = (1-1/2*sqrt2)*b3; b1 + (2-1*sqrt2)*b2 = (2-1*sqrt2)*b3; b1 + 2*b2 = 2*b3 + b4; b1 + 2*b2 = 2*b3 |
| respic | respid | point (0) | 576 | 384: 48x12v(8x3+6x4) + 192x9v(8x3+3x4) + 144x8v(8x3+2x4) | [0.666667, 0.0, 0.333333, 0.0] | b2 = 0; b4 = 0; b1 = (1+2*sqrt2)*b2 + 2*b3 + b4; b1 = 2*b3 + b4; b1 = 2*b3; b1 + 2*b2 = 2*b3 + b4; b1 + 2*b2 = 2*b3 |
| recont | redeca | point (0) | 576 | 336: 48x36v(32x3+6x8) + 288x4v(4x3) | [0.226541, 0.0, 0.773459, 0.0] | b2 = 0; b4 = 0; b1 = (1-1/2*sqrt2)*b3 + b4; b1 = (1-1/2*sqrt2)*b3; b1 + (1-1/2*sqrt2)*b2 = (1-1/2*sqrt2)*b3 + b4; b1 + (1-1/2*sqrt2)*b2 = (1-1/2*sqrt2)*b3; b1 + (1-1/2*sqrt2)*b2 + 1/2*sqrt2*b4 = (1-1/2*sqrt2)*b3 |

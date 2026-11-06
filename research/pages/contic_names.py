"""The accepted names of the F4 (icoic / contic) family's shapes, from the lists the project was given: name ->
(full name, the other symmetries it is also isogonal under)."""

ICOIC = {
    "ico": ("Icositetrachoron", "Kiicoic, Icopyritic, Toxitic, Tessic, Kitessic, Demitessic, Kidemitessic, Tessipyritic"),
    "rico": ("Rectified Icositetrachoron", "Kiicoic, Icopyritic, Toxitic, Tessic, Kitessic, Demitessic, Kidemitessic, Tessipyritic"),
    "sidpii": ("Small Disprismatoicositetricositetrachoron", "Spic"),
    "tico": ("Truncated Icositetrachoron", "Kiicoic, Icopyritic, Tessic, Kitessic, Demitessic"),
    "srico": ("Small Rhombated Icositetrachoron", "Kiicoic, Icopyritic, Toxitic"),
    "icoi": ("Icositetricositetrachoron", "Cont"),
    "grico": ("Great Rhombated Icositetrachoron", "Kiicoic, Icopyritic"),
    "prico": ("Prismatorhombated Icositetrachoron", "Kiicoic, Icopyritic"),
    "gidpii": ("Great Disprismatoicositetricositetrachoron", "Gippic"),
}
CONTIC = {
    "bicont": ("Bitetracontoctachoron", "Kicontic, Iocontic, Oxitic"),
    "spic": ("Small Prismatotetracontoctachoron", "Icoic, Kiicoic, Kicontic, Iocontic, Icopyritic, Toxitic, Oxitic"),
    "bamic": ("Biambo48", "Kicontic, Iocontic, Oxitic"),
    "cont": ("Tetracontoctachoron", "Kicontic, Iocontic, Icoic, Kiicoic, Icopyritic, Toxitic, Oxitic"),
    "bipec": ("Biprismato48", "Kicontic, Iocontic, Oxitic"),
    "bitec": ("Bitruncato48", "Kicontic, Iocontic"),
    "bimtec": ("Bimesotruncato48", "Kicontic, Iocontic, Oxitic"),
    "sabric": ("Small Birhombato48", "Kicontic, Iocontic, Oxitic"),
    "mabric": ("Medial Birhombato48", "Kicontic, Iocontic, Oxitic"),
    "gabric": ("Great Birhombato48", "Kicontic, Iocontic, Oxitic"),
    "respic": ("Rectified Spic", "Kicontic, Iocontic, Oxitic"),
    "recont": ("Rectified Cont", "Kicontic, Iocontic, Oxitic"),
    "gippic": ("Great Prismatotetracontoctachoron", "Icoic, Iocontic"),
    "sabiparc": ("Small Biprismatorhombato48", "Kicontic, Iocontic"),
    "mabiparc": ("Medial Biprismatorhombato48", "Kicontic, Iocontic"),
    "gabiparc": ("Great Biprismatorhombato48", "Kicontic, Iocontic"),
    "tabiparc": ("Transitional Biprismatorhombato48", "Kicontic, Iocontic"),
    "tispic": ("Truncated Spic", "Kicontic, Iocontic"),
    "sobcotic": ("Small Bicantitruncato48", "Kicontic, Iocontic"),
    "mobcotic": ("Medial Bicantitruncato48", "Kicontic, Iocontic"),
    "gobcotic": ("Great Bicantitruncato48", "Kicontic, Iocontic"),
    "tabcatic": ("Transitional Bicantitruncato48", "Kicontic, Iocontic"),
    "ticont": ("Truncated Cont", "Kicontic, Iocontic"),
    "sabotic": ("Small Biomnitruncato48", ""),
    "gabotic": ("Great Biomnitruncato48", ""),
    "catbotic": ("Cubic Transitional Biomnitruncato48", ""),
    "pitbotic": ("Prismatic Transitional Biomnitruncato48", ""),
}
KICONTIC = {
    "snoc": ("Snub Tetracontoctachoron", "Kiicoic, Oxitic"),
    "sabosc": ("Small Biomnisnub 48", ""), "mabosc": ("Medial Biomnisnub 48", ""), "gabosc": ("Great Biomnisnub 48", ""),
    "apbosc": ("Antipodial Biomnisnub 48", ""), "sigawbosc": ("Small Gyrowedged Biomnisnub 48", ""),
    "gigawbosc": ("Great Gyrowedged Biomnisnub 48", ""), "catbosc": ("Cubic Transitional Biomnisnub 48", ""),
    "petbosc": ("Prismatic Transitional Biomnisnub 48", ""),
}
IOCONTIC = {
    "sosbic": ("Small Omnisnub Bi48", ""), "gosbic": ("Great Omnisnub Bi48", ""),
    "catosbic": ("Cubic Transitional Omnisnub Bi48", ""), "petosbic": ("Prismatic Transitional Omnisnub Bi48", ""),
    "gippic": ("Great Prismatotetracontoctachoron", "Icoic, Contic"),
}
ICOPYRITIC = {
    "sadi": ("Snub Disicosatetrachoron", "Tessipyritic, Kidemitessic, Toxitic"),
    "prissi": ("Prismatorhombisnub Icosatetrachoron", "Toxitic"),
    "pysni": ("Pyritosnub Icositetrachoron", "a lower-symmetry srico"),
    "pysnapi": ("Pyritosnub Prismatoicositetrachoron", "a lower-symmetry prico"),
}
TOXITIC = {"sebtidi": ("Snub Bitetrahedral Dis24", "a lower-symmetry sebtic")}
OXITIC = {
    "bitac": ("Bitetrahedral 48", ""), "sebtic": ("Snub Bitetrahedral 48", "Toxitic; under toxitic symmetry it is sebtidi"),
    "sabertetic": ("Small Birhombitetratetrahedral 48", "a lower-symmetry sabric"),
    "mabertetic": ("Medial Birhombitetratetrahedral 48", "a lower-symmetry mabric"),
    "gabertetic": ("Great Birhombitetratetrahedral 48", "a lower-symmetry gabric"),
    "totabertetic": ("Triangular Transitional Birhombitetratetrahedral 48", "a lower-symmetry respic"),
    "tetabertetic": ("Tetrahedral Transitional Birhombitetratetrahedral 48", "a lower-symmetry recont"),
    "sabtutic": ("Small Bitruncatotetrahedral 48", ""), "sosbitic": ("Small Omnisnub Bitetrahedral 48", ""),
    "gosbitic": ("Great Omnisnub Bitetrahedral 48", ""),
    "stitosbetic": ("Small Triangular Transitional Omnisnub Bitetrahedral 48", ""),
    "awosbitac": ("Antiwedged Omnisnub Bitetrahedral 48", ""), "gywosbitic": ("Gyrowedged Omnisnub Bitetrahedral 48", ""),
    "sidwosbitac": ("Small Disphenoidal Omnisnub Bitetrahedral 48", ""),
    "medwosbitac": ("Medial Disphenoidal Omnisnub Bitetrahedral 48", ""),
    "gidwosbitac": ("Great Disphenoidal Omnisnub Bitetrahedral 48", ""),
    "gitatosbitac": ("Great Triangular Transitional Omnisnub Bitetrahedral 48", ""),
    "tatbitutic": ("Triangular Transitional Bitruncatotetrahedral 48", ""),
    "gabtutic": ("Great Bitruncatotetrahedral 48", ""),
    "mitatosbitac": ("Medial Triangular Transitional Omnisnub Bitetrahedral 48", ""),
    "otosbitac": ("Octahedral transitional Omnisnub Bitetrahedral 48", ""),
    "mosbitic": ("Medial Omnisnub Bitetrahedral 48", ""),
    "gadwosbitac": ("Grand Disphenoidal Omnisnub Bitetrahedral 48", ""),
}

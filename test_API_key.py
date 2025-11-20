# import os
# print(os.getenv("OPENAI_API_KEY"))

from rdkit import Chem
from rdkit.Chem import Draw

from chem_tools import make_linker_image
img = make_linker_image("c1ccccc1")  # benzene
img.show()
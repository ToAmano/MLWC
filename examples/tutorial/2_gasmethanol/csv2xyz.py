import shutil
import os
import pandas as pd
import rdkit.Chem
import rdkit.Chem.AllChem
# read csv
input_file:str = "methanol.csv" # read csv
poly = pd.read_csv(input_file)
print(" --------- ")
print(poly)
print(" --------- ")
# read smiles
smiles:str = poly["Smiles"].to_list()[0]
molname:str = poly["Name"].to_list()[0]
# build molecule
mol=rdkit.Chem.MolFromSmiles(smiles)
print(mol)
molH = rdkit.Chem.AddHs(mol) # add hydrogen
print(molH)
print(rdkit.Chem.MolToMolBlock(molH, includeStereo=False))

rdkit.Chem.AllChem.EmbedMolecule(molH, rdkit.Chem.AllChem.ETKDGv3())
print(molH)
print(rdkit.Chem.MolToMolBlock(molH, includeStereo=False))
rdkit.Chem.MolToMolFile(molH,'methanol.mol')
rdkit.Chem.MolToXYZFile(molH,'methanol.xyz')
# xyz = rdkit.Chem.MolToXYZBlock(molH)


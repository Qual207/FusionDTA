from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, AllChem
import numpy as np
RDLogger.DisableLog('rdApp.warning')

#Convert SMILES string to descriptor vector (size 200)
def featurize_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(200)

    desc_list = [
        Descriptors.MolWt(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumRotatableBonds(mol)
    ]

    #Morgan fingerprint as bit vector
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=195)
    fp_array = np.array(fp)
    desc_list.extend(fp_array.tolist())

    return np.array(desc_list, dtype=np.float32)

#Extract stereochemistry tokens from SMILES
def extract_stereo_tokens(smiles):
    stereo_tokens = []
    i = 0
    while i < len(smiles):
        atom = smiles[i]
        if atom.isalpha():
            j = i+1
            token = ''
            while j < len(smiles) and smiles[j] in ['@','/','\\']:
                token += smiles[j]
                j += 1
            if token:
                #add atom context
                stereo_tokens.append(atom + token)  
            i = j
        else:
            i += 1
    return stereo_tokens



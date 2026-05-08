import pickle
import os
from pathlib import Path
from rdkit import Chem

def build_shape_to_motifs():
    output_dir = Path("processing/output/vocabularies")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_dir / "shape_vocab.pkl", "rb") as f:
        shape_vocab = pickle.load(f)
    
    with open(output_dir / "motif_vocab.pkl", "rb") as f:
        motif_vocab = pickle.load(f)
        
    shape_to_motifs = {}
    
    for shape_idx, (shape_hash, shape_info) in enumerate(shape_vocab.items()):
        valid_motifs = []
        for motif_smiles in shape_info['motifs']:
            mol = Chem.MolFromSmiles(motif_smiles)
            if mol is None:
                continue
                
            atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
            bonds = []
            for bond in mol.GetBonds():
                bonds.append({
                    'u': bond.GetBeginAtomIdx(),
                    'v': bond.GetEndAtomIdx(),
                    'type': bond.GetBondTypeAsDouble()
                })
            
            valid_motifs.append({
                'smiles': motif_smiles,
                'atoms': atoms,
                'bonds': bonds
            })
            
        shape_to_motifs[shape_idx] = valid_motifs
        
    with open(output_dir / "shape_to_motifs.pkl", "wb") as f:
        pickle.dump(shape_to_motifs, f)
        
    print(f"Built shape_to_motifs dict with {len(shape_to_motifs)} shapes.")

if __name__ == "__main__":
    build_shape_to_motifs()

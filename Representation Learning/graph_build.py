import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data


def atom_features(atom):
    atomic_num = atom.GetAtomicNum()
    degree = atom.GetDegree()
    formal_charge = atom.GetFormalCharge()
    aromatic = int(atom.GetIsAromatic())
    hybridization = int(atom.GetHybridization())
    num_h = atom.GetTotalNumHs()
    valence = atom.GetImplicitValence()
    in_ring = int(atom.IsInRing())
    return [atomic_num, degree, formal_charge, aromatic, hybridization, num_h, valence, in_ring]


def bond_features(bond):
    bond_type = bond.GetBondTypeAsDouble()
    conjugated = int(bond.GetIsConjugated())
    in_ring = int(bond.IsInRing())
    return [bond_type, conjugated, in_ring]


def smiles_to_graph(smiles, targets):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    x_list = []
    for atom in mol.GetAtoms():
        x_list.append(atom_features(atom))
    x = torch.tensor(x_list, dtype=torch.float)
    edge_index_list = []
    edge_attr_list = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index_list.append([i, j])
        edge_index_list.append([j, i])
        bf = bond_features(bond)
        edge_attr_list.append(bf)
        edge_attr_list.append(bf)
    if len(edge_index_list) == 0:
        return None
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
    y = torch.tensor(targets, dtype=torch.float).view(1, -1)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def build_data_list(df, smiles_col="smiles", target_cols=None):
    if target_cols is None:
        target_cols = ["mu", "u0_atom", "homo", "lumo", "gap", "alpha", "cv"]
    target_cols = [c for c in target_cols if c in df.columns]
    data_list = []
    valid_indices = []
    for i in range(len(df)):
        row = df.iloc[i]
        smiles = row[smiles_col]
        targets = [float(row[c]) for c in target_cols]
        if len(targets) != len(target_cols):
            continue
        graph = smiles_to_graph(smiles, targets)
        if graph is not None:
            data_list.append(graph)
            valid_indices.append(i)
    return data_list, np.array(valid_indices)

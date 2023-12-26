'''
modify from
https://github.com/deepchem/deepchem/blob/master/deepchem/feat/molecule_featurizers/mol_graph_conv_featurizer.py#L1-L233
'''

from typing import List, Tuple
import numpy as np
import argparse
from deepchem.utils.typing import RDKitAtom, RDKitBond, RDKitMol
from deepchem.feat.graph_data import GraphData
from deepchem.feat.base_classes import MolecularFeaturizer
from deepchem.utils.molecule_feature_utils import one_hot_encode
from deepchem.utils.molecule_feature_utils import get_atom_type_one_hot
from deepchem.utils.molecule_feature_utils import construct_hydrogen_bonding_info
from deepchem.utils.molecule_feature_utils import get_atom_hydrogen_bonding_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_hybridization_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_total_num_Hs_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_is_in_aromatic_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_chirality_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_formal_charge
from deepchem.utils.molecule_feature_utils import get_atom_partial_charge
from deepchem.utils.molecule_feature_utils import get_atom_total_degree_one_hot
from deepchem.utils.molecule_feature_utils import get_bond_type_one_hot
from deepchem.utils.molecule_feature_utils import get_bond_is_in_same_ring_one_hot
from deepchem.utils.molecule_feature_utils import get_bond_is_conjugated_one_hot
from deepchem.utils.molecule_feature_utils import get_bond_stereo_one_hot

# from deepchem.utils.molecule_feature_utils import get_atom_formal_charge_one_hot
# from deepchem.utils.molecule_feature_utils import get_atom_implicit_valence_one_hot
# from deepchem.utils.molecule_feature_utils import get_atom_explicit_valence_one_hot
# from deepchem.utils.rdkit_utils import compute_all_pairs_shortest_path
# from deepchem.utils.rdkit_utils import compute_pairwise_ring_info

DEFAULT_ATOM_TYPE_SET = [
    "C",
    "N",
    "O",
    "F",
    "P",
    "S",
    "Cl",
    "Br",
    "I",
]
DEFAULT_HYBRIDIZATION_SET = ["SP", "SP2", "SP3"]
DEFAULT_TOTAL_NUM_Hs_SET = [0, 1, 2, 3, 4]
DEFAULT_FORMAL_CHARGE_SET = [-2, -1, 0, 1, 2]
DEFAULT_TOTAL_DEGREE_SET = [0, 1, 2, 3, 4, 5]
DEFAULT_RING_SIZE_SET = [3, 4, 5, 6, 7, 8]
DEFAULT_BOND_TYPE_SET = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
DEFAULT_BOND_STEREO_SET = ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"]
DEFAULT_GRAPH_DISTANCE_SET = [1, 2, 3, 4, 5, 6, 7]
DEFAULT_ATOM_IMPLICIT_VALENCE_SET = [0, 1, 2, 3, 4, 5, 6]
DEFAULT_ATOM_EXPLICIT_VALENCE_SET = [1, 2, 3, 4, 5, 6]

USER_ATOM_TYPE_SET = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
                      'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag',
                      'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni',
                      'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb']
USER_TOTAL_DEGREE_SET = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
USER_HYBRIDIZATION_SET = ["SP", "SP2", "SP3", 'SP3D', 'SP3D2']


def get_atom_implicit_valence_one_hot(
        atom: RDKitAtom,
        allowable_set: List[int] = DEFAULT_ATOM_IMPLICIT_VALENCE_SET,
        include_unknown_set: bool = True) -> List[float]:
    """Get an one-hot feature of implicit valence of an atom.
    Parameters
    ---------
    atom: rdkit.Chem.rdchem.Atom
      RDKit atom object
    allowable_set: List[int]
      Atom implicit valence to consider. The default set is `[0, 1, ..., 6]`
    include_unknown_set: bool, default True
      If true, the index of all types not in `allowable_set` is `len(allowable_set)`.
    Returns
    -------
    List[float]
      A one-hot vector of implicit valence an atom has.
      If `include_unknown_set` is False, the length is `len(allowable_set)`.
      If `include_unknown_set` is True, the length is `len(allowable_set) + 1`.
    """
    return one_hot_encode(atom.GetImplicitValence(), allowable_set,
                          include_unknown_set)


def get_atom_explicit_valence_one_hot(
        atom: RDKitAtom,
        allowable_set: List[int] = DEFAULT_ATOM_EXPLICIT_VALENCE_SET,
        include_unknown_set: bool = True) -> List[float]:
    """Get an one-hot feature of explicit valence of an atom.
    Parameters
    ---------
    atom: rdkit.Chem.rdchem.Atom
      RDKit atom object
    allowable_set: List[int]
      Atom explicit valence to consider. The default set is `[1, ..., 6]`
    include_unknown_set: bool, default True
      If true, the index of all types not in `allowable_set` is `len(allowable_set)`.
    Returns
    -------
    List[float]
      A one-hot vector of explicit valence an atom has.
      If `include_unknown_set` is False, the length is `len(allowable_set)`.
      If `include_unknown_set` is True, the length is `len(allowable_set) + 1`.
    """
    return one_hot_encode(atom.GetExplicitValence(), allowable_set,
                          include_unknown_set)


def _construct_atom_feature(
        atom: RDKitAtom, h_bond_infos: List[Tuple[int, str]], use_chirality: bool,
        use_partial_charge: bool) -> np.ndarray:
    """Construct an atom feature from a RDKit atom object.
    Parameters
    ----------
    atom: rdkit.Chem.rdchem.Atom
      RDKit atom object
    h_bond_infos: List[Tuple[int, str]]
      A list of tuple `(atom_index, hydrogen_bonding_type)`.
      Basically, it is expected that this value is the return value of
      `construct_hydrogen_bonding_info`. The `hydrogen_bonding_type`
      value is "Acceptor" or "Donor".
    use_chirality: bool
      Whether to use chirality information or not.
    use_partial_charge: bool
      Whether to use partial charge data or not.
    Returns
    -------
    np.ndarray
      A one-hot vector of the atom feature.
      44+1+5+2+1+12+6+8+7+1+1+2+1 = 91 features
    """

    # atom_type = get_atom_type_one_hot(atom,USER_ATOM_TYPE_SET,include_unknown_set = True)
    atom_type = get_atom_type_one_hot(atom, include_unknown_set=True)  # 44

    formal_charge = get_atom_formal_charge(atom)  # 1
    # hybridization = get_atom_hybridization_one_hot(atom,USER_HYBRIDIZATION_SET,include_unknown_set = False)#5
    hybridization = get_atom_hybridization_one_hot(atom, include_unknown_set=False)  # 2

    acceptor_donor = get_atom_hydrogen_bonding_one_hot(atom, h_bond_infos)  # 1
    aromatic = get_atom_is_in_aromatic_one_hot(atom)  # 12
    # degree = get_atom_total_degree_one_hot(atom,USER_TOTAL_DEGREE_SET,include_unknown_set = True)#6
    degree = get_atom_total_degree_one_hot(atom, include_unknown_set=True)

    total_num_Hs = get_atom_total_num_Hs_one_hot(atom, DEFAULT_TOTAL_NUM_Hs_SET, include_unknown_set=True)
    atom_feat = np.concatenate([
        atom_type, formal_charge, hybridization, acceptor_donor, aromatic, degree,
        total_num_Hs
    ])

    ## user additional features ####
    if True:
        imp_valence = get_atom_implicit_valence_one_hot(atom, DEFAULT_ATOM_IMPLICIT_VALENCE_SET,
                                                        include_unknown_set=True)
        exp_valence = get_atom_explicit_valence_one_hot(atom, DEFAULT_ATOM_EXPLICIT_VALENCE_SET,
                                                        include_unknown_set=True)
        atom_feat = np.concatenate([atom_feat, imp_valence, exp_valence,
                                    [atom.HasProp('_ChiralityPossible'), atom.GetNumRadicalElectrons()], ])
    ##########    END    ############     #17

    if use_chirality:
        # chirality = get_atom_chirality_one_hot(atom)
        chirality = get_atom_chirality_one_hot(atom)
        atom_feat = np.concatenate([atom_feat, np.array(chirality)])  # 2

    if use_partial_charge:
        partial_charge = get_atom_partial_charge(atom)
        atom_feat = np.concatenate([atom_feat, np.array(partial_charge)])  # 1
    return atom_feat


def _construct_bond_feature(bond: RDKitBond) -> np.ndarray:
    """Construct a bond feature from a RDKit bond object.
    Parameters
    ---------
    bond: rdkit.Chem.rdchem.Bond
      RDKit bond object
    Returns
    -------
    np.ndarray
      A one-hot vector of the bond feature.
    """
    bond_type = get_bond_type_one_hot(bond)
    same_ring = get_bond_is_in_same_ring_one_hot(bond)
    conjugated = get_bond_is_conjugated_one_hot(bond)
    stereo = get_bond_stereo_one_hot(bond)
    return np.concatenate([bond_type, same_ring, conjugated, stereo])


class MolGraphConvFeaturizer(MolecularFeaturizer):
    """This class is a featurizer of general graph convolution networks for molecules.
    The default node(atom) and edge(bond) representations are based on
    `WeaveNet paper <https://arxiv.org/abs/1603.00856>`_. If you want to use your own representations,
    you could use this class as a guide to define your original Featurizer. In many cases, it's enough
    to modify return values of `construct_atom_feature` or `construct_bond_feature`.
    The default node representation are constructed by concatenating the following values,
    and the feature length is 30.
    - Atom type: A one-hot vector of this atom, "C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "other atoms".
    - Formal charge: Integer electronic charge.
    - Hybridization: A one-hot vector of "sp", "sp2", "sp3".
    - Hydrogen bonding: A one-hot vector of whether this atom is a hydrogen bond donor or acceptor.
    - Aromatic: A one-hot vector of whether the atom belongs to an aromatic ring.
    - Degree: A one-hot vector of the degree (0-5) of this atom.
    - Number of Hydrogens: A one-hot vector of the number of hydrogens (0-4) that this atom connected.
    - Chirality: A one-hot vector of the chirality, "R" or "S". (Optional)
    - Partial charge: Calculated partial charge. (Optional)
    The default edge representation are constructed by concatenating the following values,
    and the feature length is 11.
    - Bond type: A one-hot vector of the bond type, "single", "double", "triple", or "aromatic".
    - Same ring: A one-hot vector of whether the atoms in the pair are in the same ring.
    - Conjugated: A one-hot vector of whether this bond is conjugated or not.
    - Stereo: A one-hot vector of the stereo configuration of a bond.
    If you want to know more details about features, please check the paper [1]_ and
    utilities in deepchem.utils.molecule_feature_utils.py.
    Examples
    --------
    >>> smiles = ["C1CCC1", "C1=CC=CN=C1"]
    >>> featurizer = MolGraphConvFeaturizer(use_edges=True)
    >>> out = featurizer.featurize(smiles)
    >>> type(out[0])
    <class 'deepchem.feat.graph_data.GraphData'>
    >>> out[0].num_node_features
    30
    >>> out[0].num_edge_features
    11
    References
    ----------
    .. [1] Kearnes, Steven, et al. "Molecular graph convolutions: moving beyond fingerprints."
       Journal of computer-aided molecular design 30.8 (2016):595-608.
    Note
    ----
    This class requires RDKit to be installed.
    """

    def __init__(self,
                 use_edges: bool = False,
                 use_chirality: bool = False,
                 use_partial_charge: bool = False):
        """
        Parameters
        ----------
        use_edges: bool, default False
          Whether to use edge features or not.
        use_chirality: bool, default False
          Whether to use chirality information or not.
          If True, featurization becomes slow.
        use_partial_charge: bool, default False
          Whether to use partial charge data or not.
          If True, this featurizer computes gasteiger charges.
          Therefore, there is a possibility to fail to featurize for some molecules
          and featurization becomes slow.
        """
        self.use_edges = use_edges
        self.use_partial_charge = use_partial_charge
        self.use_chirality = use_chirality

    def _featurize(self, datapoint: RDKitMol, **kwargs) -> GraphData:
        """Calculate molecule graph features from RDKit mol object.
        Parameters
        ----------
        datapoint: rdkit.Chem.rdchem.Mol
          RDKit mol object.
        Returns
        -------
        graph: GraphData
          A molecule graph with some features.
        """
        assert datapoint.GetNumAtoms(
        ) > 1, "More than one atom should be present in the molecule for this featurizer to work."
        if 'mol' in kwargs:
            datapoint = kwargs.get("mol")
            raise DeprecationWarning(
                'Mol is being phased out as a parameter, please pass "datapoint" instead.'
            )

        if self.use_partial_charge:
            try:
                datapoint.GetAtomWithIdx(0).GetProp('_GasteigerCharge')
            except:
                # If partial charges were not computed
                try:
                    from rdkit.Chem import AllChem
                    AllChem.ComputeGasteigerCharges(datapoint)
                except ModuleNotFoundError:
                    raise ImportError("This class requires RDKit to be installed.")

        # construct atom (node) feature
        h_bond_infos = construct_hydrogen_bonding_info(datapoint)
        atom_features = np.asarray(
            [
                _construct_atom_feature(atom, h_bond_infos, self.use_chirality,
                                        self.use_partial_charge)
                for atom in datapoint.GetAtoms()
            ],
            dtype=float,
        )

        # construct edge (bond) index
        src, dest = [], []
        for bond in datapoint.GetBonds():
            # add edge list considering a directed graph
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            src += [start, end]
            dest += [end, start]

        # construct edge (bond) feature
        bond_features = None  # deafult None
        if self.use_edges:
            features = []
            for bond in datapoint.GetBonds():
                features += 2 * [_construct_bond_feature(bond)]
            bond_features = np.asarray(features, dtype=float)

        return GraphData(
            node_features=atom_features,
            edge_index=np.asarray([src, dest], dtype=int),
            edge_features=bond_features)


import os
import deepchem as dc
import numpy
from rdkit import Chem
from tqdm import tqdm
import numpy as np
import hickle as hkl
import utils
from tqdm import trange
import pandas as pd
import codecs
from subword_nmt.apply_bpe import BPE


def _drug2emb_encoder(smile):
    # vocab_path = "{}/ESPF/drug_codes_chembl_freq_1500.txt".format(self.vocab_dir)
    # sub_csv = pd.read_csv("{}/ESPF/subword_units_map_chembl_freq_1500.csv".format(self.vocab_dir))
    vocab_path = "ESPF/drug_codes_chembl_freq_1500.txt"
    sub_csv = pd.read_csv("ESPF/subword_units_map_chembl_freq_1500.csv")
    bpe_codes_drug = codecs.open(vocab_path)
    dbpe = BPE(bpe_codes_drug, merges=-1, separator='')

    idx2word_d = sub_csv['index'].values
    words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))

    max_d = 50
    t1 = dbpe.process_line(smile).split()  # split
    try:
        i1 = np.asarray([words2idx_d[i] for i in t1])  # index
    except:
        i1 = np.array([0])

    l = len(i1)
    if l < max_d:
        i = np.pad(i1, (0, max_d-l), 'constant', constant_values=0)
        input_mask = ([1] * l)+([0] * (max_d-l))
    else:
        i = i1[:max_d]
        input_mask = [1] * max_d

    return i, np.asarray(input_mask)
parser = argparse.ArgumentParser(description='Drug_response_pre')
parser.add_argument('-use_aug', dest='use_aug', type=bool,default=False, help='use data aug')
parser.add_argument('-aug_num', dest='aug_num', type=bool,default=2, help='data augtimes')

args = parser.parse_args()

if __name__  == '__main__':
    count_aug=0
    drug_smiles_file = '../data/223drugs_pubchem_smiles.txt'
    pubchemid2smile = {item.split('\t')[0]: item.split('\t')[1].strip() for item in open(drug_smiles_file).readlines()}
    if (args.use_aug):
        save_dir = '../data/GDSC/aug_data_'+str(args.aug_num)
        print(save_dir)
        pubchemid2smile = {item.split('\t')[0]: item.split('\t')[1].strip() for item in
                           open(drug_smiles_file).readlines()}
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        molecules = []
        count_drug=0
        for each in tqdm(pubchemid2smile.keys()):
            if each == "84691":
                continue
            count_drug+=1
            smiles = (pubchemid2smile[each])
            if args.use_aug:
                aug_smiles = []
                for i in range(args.aug_num):
                    aug_smiles.append(Chem.MolToSmiles(Chem.MolFromSmiles(smiles), doRandom=True))
                aug_smiles.append(smiles)
                for idx, smiles in enumerate(aug_smiles):
                    save_id = each
                    if idx != len(aug_smiles) - 1:
                        save_id = "k_" + str(each) + ":" + str(count_aug)
                        count_aug += 1
                    molecules.append(Chem.MolFromSmiles(smiles))
                    graph_featurizer = MolGraphConvFeaturizer(use_edges=True, use_chirality=True,
                                                              use_partial_charge=True)
                    graph_mols = graph_featurizer.featurize(molecules)
                    node_features = graph_mols[0].node_features
                    edges_attr2 = graph_mols[0].edge_features
                    edges_index = graph_mols[0].edge_index
                    num_nodes = node_features.shape[0]
                    adj_np = np.zeros((num_nodes, num_nodes, edges_attr2.shape[1]))
                    adj_np_01 = np.zeros((num_nodes, num_nodes, edges_attr2.shape[1]))
                    index = 0
                    for i in range(0, len(edges_index[0])):
                        adj_np[edges_index[0][i]][edges_index[1][i]] = edges_attr2[index]
                        adj_np_01[edges_index[0][i]][edges_index[1][i]] = 1
                        index += 1
                    smiles_feature = _drug2emb_encoder(smiles)
                    features_tmp = [node_features, adj_np, adj_np_01, smiles_feature]
                    molecules = []
                    hkl.dump(features_tmp, '%s/%s.hkl' % (save_dir, save_id))
        print("%d drug and %d Data augmentation drugs features generated" % (count_drug, count_aug))
    else:
        count_drug=0
        save_dir = '../data/GDSC/50 and 11'
        pubchemid2smile = {item.split('\t')[0]: item.split('\t')[1].strip() for item in
                           open(drug_smiles_file).readlines()}
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        molecules = []
        for each in tqdm(pubchemid2smile.keys()):
            if each == "84691":
                continue
            count_drug+=1
            smiles = (pubchemid2smile[each])
            molecules = []
            molecules.append(Chem.MolFromSmiles(pubchemid2smile[each]))
            graph_featurizer = MolGraphConvFeaturizer(use_edges=True, use_chirality=True, use_partial_charge=True)
            graph_mols = graph_featurizer.featurize(molecules)

            node_features = graph_mols[0].node_features
            edges_attr2 = graph_mols[0].edge_features
            edges_index = graph_mols[0].edge_index
            num_nodes = node_features.shape[0]

            adj_np = np.zeros((num_nodes, num_nodes, edges_attr2.shape[1]))
            adj_np_01 = np.zeros((num_nodes, num_nodes, edges_attr2.shape[1]))
            index = 0
            for i in range(0, len(edges_index[0])):
                adj_np[edges_index[0][i]][edges_index[1][i]] = edges_attr2[index]
                adj_np_01[edges_index[0][i]][edges_index[1][i]] = 1
                index += 1
            smiles_feature = _drug2emb_encoder(smiles)
            hkl.dump([node_features, adj_np, adj_np_01, smiles_feature], '%s/%s.hkl' % (save_dir, each))
        print("%d drug's features generated" % (count_drug))



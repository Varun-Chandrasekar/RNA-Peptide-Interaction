#Step 4

import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import optim
from torch import nn
from torch.utils.data import DataLoader, random_split, Dataset, Subset
from tqdm import tqdm
from itertools import chain
from itertools import islice
import numpy as np
import random
import os
import glob
import pickle
from torch.nn.utils.rnn import pad_sequence
from Bio import SeqIO
from Bio.PDB import PDBList, PDBParser, PPBuilder
from Bio.Seq import Seq
from Bio.PDB.Polypeptide import is_aa
from Bio.SeqUtils import seq1
from collections import defaultdict
from functools import partial
import multiprocessing
from Bio.PDB.Polypeptide import is_aa

def parse_single_fasta_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    sequence = ''.join([line.strip() for line in lines if not line.startswith(">")])
    return sequence

def parse_fasta_file(filepath):
    try:
        return [str(record.seq).strip().upper() for record in SeqIO.parse(filepath, "fasta")]
    except Exception:
        return []

# Function to gather all .fasta/.fa files recursively
def get_fasta_files_from_nested_folders(root_folder, limit=None):
    files = glob.glob(os.path.join(root_folder, "**", "*.fa*"), recursive=True)
    return list(islice(files, limit)) if limit else list(files)

# Function to load sequences from multiple files using multiprocessing
def load_fasta_sequences_parallel(file_list, num_workers=4):
    if not file_list:
        return []
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.imap_unordered(parse_fasta_file, file_list, chunksize=100)
        sequences = []
        for seq_list in tqdm(results, total=len(file_list)):
            sequences.extend(seq_list)
    return sequences

def parse_fasta_sequences(filepath):
    peptides = []
    with open(filepath, 'r') as f:
        current_seq = ''
        for line in f:
            if line.startswith('>'):
                if current_seq:
                    peptides.append(current_seq)
                    current_seq = ''
            else:
                current_seq += line.strip()
        if current_seq:
            peptides.append(current_seq)
    return peptides

fasta_files = get_fasta_files_from_nested_folders("/content/drive/MyDrive/Deep Learning/RNA-fastaFiles", limit=200000)
rna_seqs = load_fasta_sequences_parallel(fasta_files, num_workers=8)
peptide_seqs = parse_fasta_sequences("/content/drive/MyDrive/Deep Learning/peptideatlas.fasta")

#Step 5

def download_multiple_pdbs(pair_file, out_dir):
    from Bio.PDB import PDBList
    import os
    os.makedirs(out_dir, exist_ok=True)

    pdbl = PDBList()
    pdb_ids = set()

    with open(pair_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2 or parts[0].lower() == "protein":
                continue  # Skip headers or malformed lines
            prot, rna = parts
            pdb_ids.add(prot.split("_")[0].lower())

    for pdb_id in sorted(pdb_ids):
        try:
            url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
            filepath = os.path.join(out_dir, f"{pdb_id}.pdb")
            if not os.path.exists(filepath):
                import urllib.request
                urllib.request.urlretrieve(url, filepath)
        except Exception:
          print(f'Nothing')


#download_multiple_pdbs("RPI2241.txt", "./pdb_files")

def load_rpi2241_pairs(filepath):
    pairs = set()
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2 or parts[0].lower() == "protein":
                continue  # Skip headers or malformed lines
            prot, rna = parts
            pairs.add((prot.upper(), rna.upper()))
    return pairs

rpi2241_positive_pairs=load_rpi2241_pairs("/content/drive/MyDrive/Deep Learning/RPI2241.txt")


#STEP 9: Converts nucleotide residue names from .pdb files (both 1-letter and 3-letter codes) to canonical 1-letter format.
# Maps 3-letter amino acid codes from .pdb files to 1-letter codes, including uncommon residues.

RNA_MAP = {
    "ADE": "A", "CYT": "C", "GUA": "G", "URI": "U",
    "PSU": "U", "INO": "I", "GTP": "G", "OMC": "C",
    "A": "A", "C": "C", "G": "G", "U": "U"  # Handle both 1/3-letter
}

AMINO_ACID_MAP = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'ASX': 'B', 'GLX': 'Z', 'SEC': 'U', 'PYL': 'O'  # Unusual residues
}

#Step 10 This function works directly with Bio.PDB chain objects and uses the mapping dictionaries to translate structural information into sequence.

def get_chain_sequence(chain):
    """Optimized residue processing"""
    seq = []
    for residue in chain:
        if residue.id[0] != " ":  # Skip heteroatoms
            continue
        resname = residue.resname.strip().upper()
        # Use direct mapping instead of seq1()
        if resname in AMINO_ACID_MAP:
            seq.append(AMINO_ACID_MAP[resname])
        elif resname in RNA_MAP:
            seq.append(RNA_MAP[resname])

    return "".join(seq)

#Step 11


SEQUENCE_CACHE_PATH = "/content/drive/MyDrive/Deep Learning/structure_sequences_cache.pkl"
CACHE_VERSION = 1

def get_chain_type(chain):
  residues = list(chain.get_residues())
  if not residues:
    return None
    # Check first residue type (efficient heuristic)
  if is_aa(residues[0]):
    return 'protein'
  else:
    return 'rna'

def get_cached_structure_sequences(pdb_dir, cache_path):
  if os.path.exists(cache_path):
    try:
      with open(cache_path, 'rb') as f:
        data = pickle.load(f)
        if data.get('version') == CACHE_VERSION:
          return data['sequences']
    except Exception:
      pass  # Recompute on error

  structure_sequences = {}
  parser = PDBParser(QUIET=True)

  for pdb_file in os.listdir(pdb_dir):
    if not pdb_file.lower().endswith(".pdb"):
      continue
    pdb_id = os.path.splitext(pdb_file)[0].lower()
    file_path = os.path.join(pdb_dir, pdb_file)
    try:
      structure = parser.get_structure(pdb_id, file_path)
      model = next(structure.get_models())
      chains = {}

      for chain in model:
        chain_type = get_chain_type(chain)
        if not chain_type:
          continue
        seq = get_chain_sequence(chain)
        chains[chain.id] = {'type': chain_type, 'sequence': seq}

      structure_sequences[pdb_id] = chains

    except Exception as e:
      print(f"Error processing {pdb_id}: {str(e)}")
      continue

  with open(cache_path, 'wb') as f:
    pickle.dump({
        'version': CACHE_VERSION,
        'sequences': structure_sequences
        }, f)
  return structure_sequences


def generate_valid_negatives(positive_pairs, structure_chains, num_negatives):
  negative_pairs = []
  positive_set = set((p, r) for p, r in positive_pairs)
  all_pdbs = list(structure_chains.keys())

  for _ in range(num_negatives):
     # Select a random PDB with protein chains
    pdb_id = random.choice(all_pdbs)
    chains = rpi_structure_chains[pdb_id]

    protein_chains = [cid for cid, info in chains.items() if info['type'] == 'protein']
    rna_chains = [cid for cid, info in chains.items() if info['type'] == 'rna']

    if not protein_chains or not rna_chains:
      continue

    prot_chain_id = random.choice(protein_chains)
    rna_chain_id = random.choice(rna_chains)
    prot_chain =f"{pdb_id}_{prot_chain_id}"
    rna_chain = f"{pdb_id}_{rna_chain_id}"

    if (prot_chain, rna_chain) in positive_set:
      continue
    negative_pairs.append((prot_chain, rna_chain, 0))

  return negative_pairs

rpi2241_negative_pairs = generate_valid_negatives(positive_pairs=rpi2241_positive_pairs, structure_chains=rpi_structure_chains, num_negatives=len(rpi2241_positive_pairs))

fasta_negative_pairs = []
for _ in range(len(rpi2241_positive_pairs)):
    rna_seq = random.choice(rna_seqs)
    pep_seq = random.choice(peptide_seqs)
    rna_idx = random.randint(0, len(rna_seqs)-1)
    pep_idx = random.randint(0, len(peptide_seqs)-1)
    fasta_negative_pairs.append((rna_idx, pep_idx))

positive_labeled = [(p, r, 1.0) for p, r in rpi2241_positive_pairs]
negative_labeled = rpi2241_negative_pairs

all_labeled_pairs = positive_labeled + negative_labeled + fasta_negative_pairs



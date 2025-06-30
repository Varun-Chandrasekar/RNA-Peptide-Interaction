#Step 12 A custom PyTorch Dataset class that handles both .pdb-based and FASTA-based sequence pairs. Handles caching, encoding, and returns torch.Tensor inputs for model training.

class RNAPeptideDataset(Dataset):
  def __init__(self, labeled_pairs, structure_sequences, pdb_dir, fasta_rna_seqs=None, fasta_pep_seqs=None):
    self.pairs = labeled_pairs
    self.structure_sequences = structure_sequences
    self.data = []
    self.pdb_dir = pdb_dir
    self.structure_cache = {}
    self.sequence_cache = {}
    self.fasta_rna_seqs = fasta_rna_seqs if fasta_rna_seqs is not None else []
    self.fasta_pep_seqs = fasta_pep_seqs if fasta_pep_seqs is not None else []

    for item in labeled_pairs:
      if len(item) == 3: # PDB pair
        prot_chain, rna_chain, label = item
        self._process_pdb_pair(prot_chain, rna_chain, label)  # Fixed method name
      elif len(item) == 2:
        rna_idx, pep_idx = item
        self._process_fasta_pair(rna_idx, pep_idx)
  def _process_pdb_pair(self, prot_chain, rna_chain, label):
    if "_" not in prot_chain or "_" not in rna_chain:
      return None
    try:
      pdb_id = prot_chain.split("_")[0].lower()
      pep_chain_id = prot_chain.split("_")[1].upper()

      rna_parts = rna_chain.split("_")
      rna_chain_id = rna_parts[1].upper() if len(rna_parts) > 1 else ""
      # Get sequences from precomputed dict
      chains = self.structure_sequences.get(pdb_id, {})
      pep_seq = chains.get(pep_chain_id, "")
      rna_seq = chains.get(rna_chain_id, "")

      if not pep_seq or not rna_seq:
        return None

      self._add_to_data(rna_seq, pep_seq, label)

    except IndexError as e:
      print(f"Format error in chain IDs: {e}")
      print(f"prot_chain: {prot_chain}, rna_chain: {rna_chain}")

  def _process_fasta_pair(self, rna_idx, pep_idx):
    try:
      rna_seq = self.fasta_rna_seqs[rna_idx]
      pep_seq = self.fasta_pep_seqs[pep_idx]
      self._add_to_data(rna_seq, pep_seq, 0)
    except IndexError:
      return None

  def _add_to_data(self, rna_seq, pep_seq, label):
    rna_tensor = self._cached_encode(rna_seq, "RNA")
    pep_tensor = self._cached_encode(pep_seq, "PEP")
    label_tensor = torch.tensor([label], dtype=torch.float)

    self.data.append((rna_tensor, pep_tensor, label_tensor))

  def _cached_encode(self, sequence, seq_type):
    cache_key = (sequence, seq_type)
    if cache_key not in self.sequence_cache:
      if seq_type == "RNA":
        tensor = one_hot_encodeRNA(sequence).float()
      else:
        tensor = one_hot_encodepeptide(sequence).float()
      self.sequence_cache[cache_key] = tensor
    return self.sequence_cache[cache_key]

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    try:
      return self.data[idx]
    except IndexError:
      return None

def collate_fn(batch):
  # Filter None items and check for empty batch
  batch = [item for item in batch if item is not None]
  if not batch:
    return torch.tensor([]), torch.tensor([]), torch.tensor([])

  # Unpack with shape validation
  rna_seqs, pep_seqs, labels = [], [], []
  for item in batch:
    r, p, l = item
    # Validate channel dimensions
    if r.shape[-1] != 4:
      r = F.one_hot(r.argmax(-1), num_classes=4).float()  # Repair RNA
    if p.shape[-1] != 20:
      p = F.one_hot(p.argmax(-1), num_classes=20).float() # Repair PEP
    rna_seqs.append(r)
    pep_seqs.append(p)
    labels.append(l)

  # Pad sequences
  rna_padded = pad_sequence(rna_seqs, batch_first=True)
  pep_padded = pad_sequence(pep_seqs, batch_first=True)

  # Verify batch consistency
  assert len(rna_padded) == len(pep_padded) == len(labels)
  return rna_padded, pep_padded, torch.stack(labels)

# step 13:Creates the full dataset, splits into train/val/test sets using stratified logic over .pdb IDs and FASTA pairs, then wraps with DataLoaders.


full_dataset = RNAPeptideDataset(labeled_pairs=all_labeled_pairs, structure_sequences=rpi_structure_chains, pdb_dir="/content/drive/MyDrive/Deep Learning/pdb_files", fasta_rna_seqs=rna_seqs, fasta_pep_seqs=peptide_seqs)

pdb_pairs = [pair for pair in all_labeled_pairs if len(pair[0].split("_")) == 2]
fasta_pairs = [pair for pair in all_labeled_pairs if len(pair[0].split("_")) != 2]

pdb_to_indices = defaultdict(list)
for idx, (prot_chain, rna_chain, label) in enumerate(pdb_pairs):
    pdb_id = prot_chain.split("_")[0].lower()
    pdb_to_indices[pdb_id].append(idx)

pdb_ids = list(pdb_to_indices.keys())
np.random.shuffle(pdb_ids)

n_total_pdb = len(pdb_ids)
n_train_pdb = int(0.8 * n_total_pdb)
n_val_pdb = int(0.15 * n_total_pdb)
n_test_pdb = n_total_pdb - n_train_pdb - n_val_pdb

train_pdbs = pdb_ids[:n_train_pdb]
val_pdbs = pdb_ids[n_train_pdb:n_train_pdb+n_val_pdb]
test_pdbs = pdb_ids[n_train_pdb+n_val_pdb:]

train_indices_pdb = [idx for pdb in train_pdbs for idx in pdb_to_indices[pdb]]
val_indices_pdb = [idx for pdb in val_pdbs for idx in pdb_to_indices[pdb]]
test_indices_pdb = [idx for pdb in test_pdbs for idx in pdb_to_indices[pdb]]

fasta_indices = list(range(len(pdb_pairs), len(pdb_pairs) + len(fasta_pairs)))
np.random.shuffle(fasta_indices)

n_total_fasta = len(fasta_indices)
n_train_fasta = int(0.8 * n_total_fasta)
n_val_fasta = int(0.15 * n_total_fasta)
n_test_fasta = n_total_fasta - n_train_fasta - n_val_fasta

train_indices_fasta = fasta_indices[:n_train_fasta]
val_indices_fasta = fasta_indices[n_train_fasta:n_train_fasta+n_val_fasta]
test_indices_fasta = fasta_indices[n_train_fasta+n_val_fasta:]

train_indices = train_indices_pdb + train_indices_fasta
val_indices = val_indices_pdb + val_indices_fasta
test_indices = test_indices_pdb + test_indices_fasta

train_dataset = Subset(full_dataset, train_indices)
val_dataset = Subset(full_dataset, val_indices)
test_dataset = Subset(full_dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)





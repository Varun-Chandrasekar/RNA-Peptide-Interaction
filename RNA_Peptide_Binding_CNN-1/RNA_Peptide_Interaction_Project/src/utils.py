# Recursively searches for .fasta/.fa files.

def get_fasta_files_from_nested_folders(root_folder, limit=None):
    files = glob.glob(os.path.join(root_folder, "**", "*.fa*"), recursive=True)
    return list(islice(files, limit)) if limit else list(files)


# Encodes an RNA sequence into a one-hot matrix of shape [L, 4], where L is the sequence length.

# One-hot encoding function for RNA and Peptide sequences
def one_hot_encodeRNA(sequence):
    mapping = {'A': 0, 'U': 1, 'G': 2, 'C': 3}
    one_hot = np.zeros((len(sequence), 4))
    for i, base in enumerate(sequence):
        if base in mapping:
            one_hot[i, mapping[base]] = 1
    return torch.tensor(one_hot, dtype=torch.float)

# Encodes a peptide sequence into a one-hot matrix of shape [L, 20].

def one_hot_encodepeptide(sequence):
    amino_acids = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I',
    'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    aa_to_index = {aa: idx for idx, aa in enumerate(amino_acids)}
    one_hot = np.zeros((len(sequence), 20))
    for i, aa in enumerate(sequence):
        if aa in aa_to_index:
            one_hot[i, aa_to_index[aa]] = 1
    return torch.tensor(one_hot, dtype=torch.float)

# Applies a specified encoding function (e.g., one-hot) to a list of sequences and wraps the loop in a progress bar.

def precompute_fasta_encodings(seqs, encode_fn):
    return [encode_fn(seq) for seq in tqdm(seqs)]

# Returns a cached one-hot tensor for a given sequence, saving encoding time for repeated inputs. The encoder_name ensures different encoders (e.g., RNA vs. peptide) are managed independently.

ENCODING_CACHE = {}

def cached_one_hot_encode(sequence, encoder, encoder_name):
    key = (encoder_name, sequence)
    if key not in ENCODING_CACHE:
        ENCODING_CACHE[key] = encoder(sequence).float()
    return ENCODING_CACHE[key]


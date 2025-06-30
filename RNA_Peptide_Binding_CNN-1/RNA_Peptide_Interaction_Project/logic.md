# RNA–Peptide Interaction Prediction – Algorithm & Structure

This document explains the logic and structure of the deep learning pipeline implemented in this repository.

---

## Step 1 – Library Installation
Install necessary libraries:
```bash
pip install biopython torch numpy scikit-learn
```

## Step 2 – Mount Google Drive (Colab only)
Used to access `.pdb` and `.fasta` datasets stored on Google Drive.
```python
from google.colab import drive
drive.mount('/content/drive')
```

## Step 3 – Import Libraries
Standard imports include:
- torch, torch.nn, torch.optim, torch.utils.data, torch.nn.functional – Deep learning framework for model building, training, and dataset handling
- tqdm – Progress bars for loops and data loading 
- numpy – Efficient numerical computation and array operations
- BioPython – parsing and analyzing biological sequences and structures.
- pickle, os, multiprocessing – file and performance tools


## Step 4 – Sequence Extraction from FASTA Files

- `parse_single_fasta_file(path)`: Reads a single FASTA file and returns the sequence string.
- `parse_fasta_file(filepath)`: Parses a FASTA file using Biopython, returns uppercase sequences.
- `get_fasta_files_from_nested_folders(root_folder)`: Recursively finds `.fasta`/`.fa` files in nested folders.
- `load_fasta_sequences_parallel(file_list)`: Uses multiprocessing to load sequences from a list of files.
- `parse_fasta_sequences(filepath)`: Manually parses multi-sequence FASTA files (e.g., from PeptideAtlas).

These functions are located in:
- `data_processing.py`: sequence and file parsing logic
- `utils.py`: helper functions for file system navigation

Use the `notebooks/` directory to run the sequence extraction logic before model training.


## Step 5 – RPI2241 Interaction Pairs and PDB Structure Download

- `download_multiple_pdbs(pair_file, out_dir)`: Downloads PDB structure files from the RCSB database for a given set of protein–RNA interaction pairs. Parses the `.txt` file and retrieves the relevant `.pdb` files. (Note: Please remove the hash in the next line if you DO NOT want to manually download the fines)

- `load_rpi2241_pairs(filepath)`: Loads positive interaction pairs (protein, RNA) from the dataset (RPI2241). Skips headers and ensures standard uppercase formatting.

These functions are located in `data_processing.py` and are critical for preparing the input structures used for feature extraction.

Sample usage is provided in the starter notebook.

## Step 6 – Sequence Encoding

- `one_hot_encodeRNA(sequence)`: Encodes RNA sequence into a `[L, 4]` one-hot tensor, where L is the sequence length and each base is mapped to {A, U, G, C}.

- `one_hot_encodepeptide(sequence)`: Encodes peptide sequence into a `[L, 20]` one-hot tensor based on the 20 canonical amino acids.

These functions are implemented in `utils.py` and are used for preparing model-ready input tensors.

## Step 8 – Precomputing Encoded Tensors

- `precompute_fasta_encodings(seqs, encode_fn)`: Applies an encoding function like `one_hot_encodeRNA` or `one_hot_encodepeptide` to a list of sequences. Uses `tqdm` for progress tracking.

This utility is located in `utils.py`. The resulting tensors (RNA and peptide) are used as direct input to the CNN model.

## Step 9 – Residue Code Mappings

- `RNA_MAP`: A dictionary for converting both 3-letter and 1-letter RNA codes (e.g., ADE → A, GUA → G) from PDB files into canonical RNA characters. Located in `data_processing.py`.

- `AMINO_ACID_MAP`: A mapping from 3-letter to 1-letter amino acid codes, including unusual residues like SEC (selenocysteine) and PYL (pyrrolysine). Located in `utils.py`.

These mappings are used during sequence extraction and pre-encoding from structure files.


## Step 10 – Extract Sequence from PDB Chain

- `get_chain_sequence(chain)`: Converts a chain from a PDB file into its corresponding sequence. It skips non-standard residues and uses `RNA_MAP` and `AMINO_ACID_MAP` to convert 3-letter residue names to 1-letter sequences.

This function is located in `data_processing.py` and is used when parsing `.pdb` files for feature extraction.

## Step 11 – Structural Sequence Caching and Negative Pair Generation

- `get_chain_type(chain)`: Determines whether a chain is a protein or RNA based on its first residue.

- `get_cached_structure_sequences(pdb_dir, cache_path)`: Parses and caches protein/RNA sequences from `.pdb` files. If a valid pickle cache exists, it loads from there.

- `generate_valid_negatives(...)`: Creates negative training pairs by randomly sampling protein and RNA chains from different structures while avoiding overlap with positive pairs.

Runtime configuration such as file paths and combining datasets (positive + negative) are handled in the starter notebook.

## Step 12 – Encoding Cache

- `ENCODING_CACHE`: A dictionary that stores already-encoded sequences to prevent redundant computation.

- `cached_one_hot_encode(sequence, encoder, encoder_name)`: Efficiently computes or retrieves one-hot encodings using a cache keyed by `(encoder_name, sequence)`. Useful when working with repeated sequences in large datasets.

Located in `utils.py`.

## Step 13 – Dataset Construction and Splitting

- `RNAPeptideDataset`: A custom PyTorch `Dataset` that supports both PDB-derived and FASTA-derived peptide–RNA sequence pairs. Automatically caches and one-hot encodes sequences.

- `collate_fn(batch)`: Pads sequences and ensures tensor shapes are valid. Used in the `DataLoader`.

- Runtime code in the notebook splits the dataset by PDB ID and FASTA index into training, validation, and test sets. Ensures that no data leakage occurs across splits.

Located in `dataset.py` (class and collate function) and `notebooks/` (data preparation and loaders).

## Step 14 – CNN Model Architecture

- `CNN(nn.Module)`: A PyTorch neural network model with two parallel convolutional branches:
  - RNA input with shape `[B, 4, L]`
  - Peptide input with shape `[B, 20, L]`

  The outputs of both branches are pooled, concatenated, passed through fully connected layers, and output a binding score. Includes:
  - 3 layers of 1D convolutions for each branch
  - `AdaptiveAvgPool1d` to reduce sequence dependence
  - Input padding for short sequences

This model is implemented in `model.py` and can be used directly in the training or prediction scripts.

## Step 15 – Model Initialization and Training Setup

- `model = CNN()`: Instantiates the CNN model defined in `model.py`.

- `BCEWithLogitsLoss(pos_weight=...)`: Computes binary cross-entropy with a class weighting factor to account for imbalance between positive and negative interaction pairs.

- `Adam Optimizer`: Used to update model weights based on gradients with a learning rate of 0.0005.

These components are implemented during training in `train.py` or interactively in the starter notebook.

## Step 16 – Model Training Loop with GPU and Validation

- Uses PyTorch's `cuda` support to train on GPU if available.
- For each epoch:
  - Model is set to training mode with `model.train()`
  - Batch data is sent to the appropriate device
  - Gradients are zeroed, forward/backward passes are run, and weights are updated
  - Training loss is computed and stored

- After each epoch:
  - `model.eval()` switches to evaluation mode
  - No gradient updates during validation (`torch.no_grad()`)
  - Average validation loss is computed

Implemented in `train.py`.

## Step 17 – Saving the Trained Model

- `torch.save(model.state_dict(), "models/cnn.pt")`: Saves the trained model weights to disk for future inference or fine-tuning.

This step is typically executed at the end of training inside `train.py`.

---

## Summary
Each functional block is modularized into `src/`. Use the starter notebook in `notebooks/` to interact with the components.


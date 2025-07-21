
This project utilizes a deep learning model to predict the interaction between RNA and peptide molecules. The model for this project is based on a dual-branch 1D CNN (with 3 convolution layers) trained on curated RNA–peptide pairs.

## Data
This project accepts .pdb files and .fasta files which can be downloaded from Peptipedia, PDB, bpRNA and RPISeq

##  Project Structure

RNA-Peptide-Binding-CNN/
├── data_processing/
│   ├── fasta_utils.py
│   ├── pdb_utils.py
│   └── sequence_utils.py
├── model/
│   └── cnn_model.py
├── datasets/
│   └── rna_peptide_dataset.py
├── training/
│   └── train.py
├── config.py
└── main.py

##  Model Overview

- Two parallel 1D CNN branches for RNA and peptide one-hot sequences
- Adaptive pooling and concatenation
- Fully connected layers with sigmoid output for binary classification

##  Dataset Construction

- Positive pairs are parsed from RPI2241.
- Negative pairs are generated from unrelated chains in `.pdb` or randomized `.fasta` sequences.
- Dataset logic implemented in `src/dataset.py`, `data_processing.py`.

---




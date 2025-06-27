
THIS PROJECT IS CURRENTLY UNDER CONSTRUCTION AND IS IN THE PROCESS OF BEING UPLOADED. THE CURRENT FOLDER ONLY CONTAINS THE FULL CODE.

This project utilizes a deep learning model (convolution neural network (CNN))) to predict the interaction between an RNA molecule and peptide molecule. More specifically, whether the two molecules would bind or not.

## Data
This project accepts .pdb files and .fasta files

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

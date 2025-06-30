import argparse
from src.train import train
from src.predict import predict

def main():
    parser = argparse.ArgumentParser(description="RNA–Peptide Interaction Pipeline")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'predict'], help="Mode: train or predict")
    parser.add_argument('--data_path', type=str, default="data/", help="Path to input data")
    parser.add_argument('--model_path', type=str, default="models/rna_peptide_cnn.pt", help="Path to save/load model")
    parser.add_argument('--epochs', type=int, default=20, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Training batch size")

    args = parser.parse_args()

    if args.mode == 'train':
        train(data_path=args.data_path, model_path=args.model_path, epochs=args.epochs, batch_size=args.batch_size)
    elif args.mode == 'predict':
        predict(model_path=args.model_path, data_path=args.data_path)

if __name__ == "__main__":
    main()

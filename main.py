from src.train import train_model
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Heart Disease (Cleveland)")
    parser.add_argument("models", nargs="*", default=["rf"],
                        help="Quali modelli addestrare: knn tree rf (default: rf)")
    parser.add_argument("--no-fe", action="store_true",
                        help="Disabilita il Feature Engineering nella pipeline")
    parser.add_argument("--age-bin", action="store_true",
                        help="Aggiunge il binning dell'età come feature categorica")
    args = parser.parse_args()

    print("Avvio addestramento…")
    train_model(
        models=args.models,
        use_fe=not args.no_fe,
        add_age_bin=args.age_bin,
    )

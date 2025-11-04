import os
import argparse
from pathlib import Path
import bin.eval
import bin.train
import bin.data
import bin.model
from datetime import datetime


def main():
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        default="dataset/",
        help="Path to dataset containing train/ and test/ folders",
    )
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--model_out", default="models/vit_v" + timestamp_str + ".pt"
    )
    args = parser.parse_args()

    print("\n===> Loading datasets")
    train_loader, val_loader, test_loader, class_names = bin.data.make_datasets(
        args.dataset_dir, args.image_size, args.batch_size
    )

    print(f"\n===> Building Vision Transformer model for {len(class_names)} classes...")
    model, base_model = bin.model.build_model(len(class_names), args.image_size)

    print("\n===> Starting training pipeline...")
    bin.train.train_model(
        model, base_model, train_loader, val_loader, args.epochs, args.model_out
    )

    print("\n===> Evaluating trained model on test dataset...")
    bin.eval.evaluate_model(
        args.model_out, args.dataset_dir, args.image_size, args.batch_size
    )

    print("\n===> Pipeline complete!")
    print(
        "\n===> Training and evaluation reports are saved in the 'results/' directory"
    )


if __name__ == "__main__":
    main()
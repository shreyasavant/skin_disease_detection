import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
from pathlib import Path
from torchvision import datasets, transforms


def load_test_dataset(dataset_dir, image_size=224, batch_size=32):
    test_dir = Path(dataset_dir) / "test"
    
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_ds = datasets.ImageFolder(root=str(test_dir), transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    class_names = test_ds.classes
    
    return test_loader, class_names


def evaluate_model(model_path, dataset_dir, image_size=224, batch_size=32):
    print(f"\n===> Loading model from: {model_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load test dataset first to get num_classes
    test_loader, class_names = load_test_dataset(dataset_dir, image_size, batch_size)
    
    # Import model architecture
    import bin.model as model_module
    model, _ = model_module.build_model(len(class_names), image_size)
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    print(f"\n===> Evaluating model on test dataset...")
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print("\n===> Evaluation Complete!")
    print(
        f"Accuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1-score: {f1:.4f}"
    )

    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\n===> Classification Report:\n", report)

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # create results folder
    os.makedirs("results", exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    pdf_path = os.path.join("results", "vit_evaluation_v" + timestamp_str + ".pdf")

    with PdfPages(pdf_path) as pdf:
        # confusion matrix heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix - Vision Transformer")
        pdf.savefig()
        plt.close()

        # metrics summary page
        plt.figure(figsize=(8.5, 11))
        plt.axis("off")
        text = (
            f"Vision Transformer Evaluation Summary\n\n"
            f"Accuracy: {acc:.4f}\n"
            f"Precision: {prec:.4f}\n"
            f"Recall: {rec:.4f}\n"
            f"F1-score: {f1:.4f}\n\n"
            f"Classification Report:\n{report}"
        )
        plt.text(0.02, 0.98, text, fontsize=10, va="top", ha="left", family="monospace")
        pdf.savefig()
        plt.close()

    print(f"\n===> Evaluation report saved to: {pdf_path}")
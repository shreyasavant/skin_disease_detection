import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime


def plot_training_curves(history, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    pdf_path = os.path.join(output_dir, "training_curves_v" + timestamp_str + ".pdf")

    with PdfPages(pdf_path) as pdf:
        # accuracy plot
        plt.figure(figsize=(8, 6))
        plt.plot(history['train_acc'], label="Train Accuracy")
        plt.plot(history['val_acc'], label="Val Accuracy")
        plt.title("Training and Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        pdf.savefig()
        plt.close()

        # loss plot
        plt.figure(figsize=(8, 6))
        plt.plot(history['train_loss'], label="Train Loss")
        plt.plot(history['val_loss'], label="Val Loss")
        plt.title("Training and Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        pdf.savefig()
        plt.close()

    print(f"\n===> Training curves saved to {pdf_path}")


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def train_model(model, base_model, train_loader, val_loader, epochs, model_out):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    patience = 6
    patience_counter = 0
    
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    
    print("\n===> Starting training...")
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_out.replace('.pt', '_best.pt'))
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n===> Early stopping at epoch {epoch+1}")
            break
    
    print("\n===> Generating training performance plots...")
    plot_training_curves(history)
    
    # Restore best model weights before fine-tuning (matching CNN behavior)
    if os.path.exists(model_out.replace('.pt', '_best.pt')):
        print("\n===> Restoring best model weights before fine-tuning...")
        model.load_state_dict(torch.load(model_out.replace('.pt', '_best.pt'), map_location=device))
        print(f"===> Restored model with validation accuracy: {best_val_acc:.2f}%")
    
    # Fine-tuning
    print("\n===> Fine-tuning the base model...")
    for param in base_model.parameters():
        param.requires_grad = True
    
    # Freeze early layers (keep last 40% trainable)
    total_layers = len(list(base_model.encoder.layer))
    freeze_until = int(total_layers * 0.6)
    for i, layer in enumerate(base_model.encoder.layer):
        if i < freeze_until:
            for param in layer.parameters():
                param.requires_grad = False
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    scheduler_ft = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    fine_tune_epochs = max(3, int(epochs * 0.3))
    
    fine_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc_ft = best_val_acc
    patience_counter_ft = 0
    
    print(f"\n===> Fine-tuning for {fine_tune_epochs} epochs...")
    for epoch in range(fine_tune_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        fine_history['train_loss'].append(train_loss)
        fine_history['train_acc'].append(train_acc)
        fine_history['val_loss'].append(val_loss)
        fine_history['val_acc'].append(val_acc)
        
        scheduler_ft.step(val_loss)
        
        # Save best fine-tuned model
        if val_acc > best_val_acc_ft:
            best_val_acc_ft = val_acc
            torch.save(model.state_dict(), model_out.replace('.pt', '_best_ft.pt'))
            patience_counter_ft = 0
        else:
            patience_counter_ft += 1
        
        # Early stopping for fine-tuning
        if patience_counter_ft >= patience:
            print(f"\n===> Early stopping fine-tuning at epoch {epoch+1}")
            break
        
        print(f"Fine-tune Epoch {epoch+1}/{fine_tune_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    # Restore best fine-tuned weights
    if os.path.exists(model_out.replace('.pt', '_best_ft.pt')):
        model.load_state_dict(torch.load(model_out.replace('.pt', '_best_ft.pt'), map_location=device))
        print(f"\n===> Restored best fine-tuned model (Val Acc: {best_val_acc_ft:.2f}%)")
    
    print("\n===> Adding fine-tuning performance to training report...")
    plot_training_curves(fine_history)
    
    # Save final model
    torch.save(model.state_dict(), model_out)
    print(f"===> Model saved to {model_out}")
    
    return history
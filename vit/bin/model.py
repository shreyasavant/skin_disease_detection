import os
import torch
import torch.nn as nn
from pathlib import Path
from dotenv import load_dotenv
from transformers import ViTModel, ViTConfig

# Load environment variables
load_dotenv()


class ViTClassifier(nn.Module):
    def __init__(self, num_classes, model_name="google/vit-base-patch16-224"):
        super().__init__()
        self.vit = ViTModel.from_pretrained(model_name)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.vit.config.hidden_size, num_classes)
        )
        
        # Freeze ViT initially
        for param in self.vit.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        outputs = self.vit(pixel_values=x)
        # Get [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0]
        logits = self.classifier(cls_output)
        return logits


def build_model(num_classes, image_size=224, model_name=None):
    if model_name is None:
        model_name = os.getenv("VIT_MODEL", "google/vit-base-patch16-224")
    
    model_dict = {
        "vit-base-patch16-224": "google/vit-base-patch16-224",
        "vit-base-patch32-224": "google/vit-base-patch32-224",
        "vit-large-patch16-224": "google/vit-large-patch16-224",
        "vit-base-patch16-384": "google/vit-base-patch16-384",
    }
    
    model_id = model_dict.get(model_name, "google/vit-base-patch16-224")
    
    model = ViTClassifier(num_classes=num_classes, model_name=model_id)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    print(f"\n===> Model built with {model_id}")
    print(f"===> Device: {device}")
    
    return model, model.vit
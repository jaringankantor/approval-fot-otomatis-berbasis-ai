import torch
import numpy as np
from PIL import Image

from torchvision import models, transforms


class ResNet50FeatureExtractor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)

        # Hilangkan layer klasifikasi terakhir.
        # Output menjadi fitur 2048 dimensi.
        self.model = torch.nn.Sequential(*list(model.children())[:-1])
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=weights.transforms().mean,
                std=weights.transforms().std
            )
        ])

    def extract(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feature = self.model(image_tensor)

        feature = feature.squeeze().cpu().numpy()
        feature = feature.astype(np.float32)

        return feature
import torch.nn as nn
import torchvision.models as models

class ReIDModel(nn.Module):
    def __init__(self, num_classes, embedding_dim=512):
        super().__init__()
        base = models.resnet50(weights="IMAGENET1K_V1")
        base.fc = nn.Identity()
        self.backbone = base
        self.embedding = nn.Linear(2048, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        embed = self.embedding(x)
        logits = self.classifier(embed)
        return embed, logits

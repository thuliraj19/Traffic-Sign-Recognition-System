"""
train_classifier.py
--------------------
Trains a traffic sign classifier on the Mapillary dataset using one of three backbones:
    - baseline_cnn : simple custom CNN (from scratch)
    - mobilenet_v2 : transfer learning (ImageNet pretrained)
    - efficientnet_b0 : transfer learning (ImageNet pretrained)

Usage:
    python train_classifier.py --backbone efficientnet_b0 --epochs 30 --batch_size 64
"""

import os
import sys
import argparse
import time
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data.augmentations import get_train_transforms, get_val_transforms

DEFAULT_NUM_CLASSES = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------
# Model definitions
# ----------------------------------------------------------------------
class BaselineCNN(nn.Module):
    """Lightweight CNN trained from scratch as the baseline comparison model."""
    def __init__(self, num_classes=DEFAULT_NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),  # 64 -> 32

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),  # 32 -> 16

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),  # 16 -> 8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128 * 8 * 8, 256), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def build_model(backbone: str, num_classes=DEFAULT_NUM_CLASSES):
    if backbone == "baseline_cnn":
        return BaselineCNN(num_classes)

    elif backbone == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
        return model

    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    else:
        raise ValueError(f"Unknown backbone: {backbone}")


# ----------------------------------------------------------------------
# Dataset wrapper (applies albumentations to torchvision ImageFolder)
# ----------------------------------------------------------------------
class AlbumentationsImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform):
        super().__init__(root)
        self.alb_transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = self.loader(path)
        image = self.alb_transform(image=self._to_numpy(image))["image"]
        return image, target

    @staticmethod
    def _to_numpy(pil_image):
        import numpy as np
        return np.array(pil_image)


# ----------------------------------------------------------------------
# Train / evaluate loops
# ----------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    for images, labels in tqdm(loader, desc="Eval", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    avg_loss = running_loss / len(all_labels)
    return avg_loss, acc, f1, all_labels, all_preds


def main(args):
    print(f"Device: {DEVICE}")
    print(f"Backbone: {args.backbone}")

    train_ds = AlbumentationsImageFolder(
        os.path.join(args.data_dir, "train"),
        get_train_transforms(img_size=args.img_size, severity=args.aug_severity),
    )
    val_ds = AlbumentationsImageFolder(
        os.path.join(args.data_dir, "val"),
        get_val_transforms(img_size=args.img_size),
    )

    class_names = train_ds.classes
    num_classes = len(class_names)
    print(f"Detected {num_classes} classes: {class_names}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = build_model(args.backbone, num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1 = 0.0
    history = []
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"[Epoch {epoch}/{args.epochs}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} "
              f"({elapsed:.1f}s)")

        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1
        })

        if val_f1 > best_f1:
            best_f1 = val_f1
            ckpt_path = os.path.join(args.output_dir, f"{args.backbone}_best.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "backbone": args.backbone,
                "num_classes": num_classes,
                "classes": class_names,
                "img_size": args.img_size,
                "val_f1": val_f1,
                "val_acc": val_acc,
            }, ckpt_path)
            print(f"  -> New best model saved: {ckpt_path} (F1={val_f1:.4f})")

    with open(os.path.join(args.output_dir, f"{args.backbone}_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val F1: {best_f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/mapillary_classification")
    parser.add_argument("--output_dir", type=str, default="models")
    parser.add_argument("--backbone", type=str, default="efficientnet_b0",
                         choices=["baseline_cnn", "mobilenet_v2", "efficientnet_b0"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--aug_severity", type=str, default="moderate",
                         choices=["light", "moderate", "heavy"])
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    main(args)

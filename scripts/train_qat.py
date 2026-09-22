import os
import sys
import argparse
import time
import json
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.ao.quantization import get_default_qat_qconfig, prepare_qat, convert
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scripts.train_classifier import build_model, AlbumentationsImageFolder
from data.augmentations import get_train_transforms, get_val_transforms

DEVICE = torch.device("cpu")
torch.backends.quantized.engine = "fbgemm"  # required before running any converted INT8 model

def compute_class_weights(dataset):
    labels = [label for _, label in dataset.samples]
    counts = Counter(labels)
    total = sum(counts.values())
    n_classes = len(counts)
    weights = torch.tensor([total / (n_classes * counts[i]) for i in range(n_classes)], dtype=torch.float32)
    print("Class distribution:", dict(sorted(counts.items())))
    print("Class weights:", weights.tolist())
    return weights

@torch.no_grad()
def evaluate(model, loader, class_names=None):
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in tqdm(loader, desc="Eval", leave=False):
        outputs = model(images)
        all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    report = classification_report(all_labels, all_preds, target_names=class_names if class_names else None, zero_division=0)
    return acc, f1, report

def main(args):
    print(f"QAT fine-tuning: {args.backbone}")
    print(f"Device: {DEVICE} (QAT runs on CPU)")
    train_ds = AlbumentationsImageFolder(os.path.join(args.data_dir, "train"), get_train_transforms(img_size=args.img_size, severity=args.aug_severity))
    val_ds = AlbumentationsImageFolder(os.path.join(args.data_dir, "val"), get_val_transforms(img_size=args.img_size))
    class_names = train_ds.classes
    num_classes = len(class_names)
    print(f"Detected {num_classes} classes: {class_names}")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = build_model(args.backbone, num_classes)
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded FP32 starting weights from {args.checkpoint} (val_f1={ckpt.get('val_f1', 'n/a')})")
    else:
        print("WARNING: no checkpoint given -- QAT from scratch is much slower to converge.")
    model.to(DEVICE)
    print("\n--- Baseline FP32 accuracy (before QAT) ---")
    base_acc, base_f1, base_report = evaluate(model, val_loader, class_names)
    print(f"FP32 Accuracy: {base_acc:.4f} | Macro F1: {base_f1:.4f}")
    model.train()
    model.qconfig = get_default_qat_qconfig(args.backend)
    print(f"\nQAT backend: {args.backend}")
    try:
        model_prepared = prepare_qat(model, inplace=False)
    except Exception as e:
        print(f"\nERROR during prepare_qat: {e}")
        print("Try --backbone baseline_cnn, which fuses more reliably.")
        raise
    if args.class_weights:
        weights = compute_class_weights(train_ds)
        criterion = nn.CrossEntropyLoss(weight=weights)
        print("Using class-weighted loss to counter dataset imbalance.")
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model_prepared.parameters(), lr=args.lr, weight_decay=1e-4)
    history = []
    os.makedirs(args.output_dir, exist_ok=True)
    best_f1 = 0.0
    for epoch in range(1, args.epochs + 1):
        model_prepared.train()
        t0 = time.time()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in tqdm(train_loader, desc=f"QAT Epoch {epoch}", leave=False):
            optimizer.zero_grad()
            outputs = model_prepared(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
        train_loss = running_loss / total
        train_acc = correct / total
        val_acc, val_f1, _ = evaluate(model_prepared, val_loader, class_names)
        elapsed = time.time() - t0
        print(f"[QAT Epoch {epoch}/{args.epochs}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} | val_acc={val_acc:.4f} val_f1={val_f1:.4f} ({elapsed:.1f}s)")
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_acc": val_acc, "val_f1": val_f1})
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({"model_state_dict": model_prepared.state_dict(), "backbone": args.backbone, "num_classes": num_classes, "img_size": args.img_size, "val_f1": val_f1, "val_acc": val_acc, "qat": True}, os.path.join(args.output_dir, f"{args.backbone}_qat.pt"))
            print(f"  -> Saved QAT checkpoint (F1={val_f1:.4f})")
    print("\n--- Converting QAT model to INT8 ---")
    model_prepared.eval()
    model_int8 = convert(model_prepared.cpu(), inplace=False)
    int8_path = os.path.join(args.output_dir, f"{args.backbone}_qat_int8.pth")
    torch.save(model_int8.state_dict(), int8_path)
    print(f"INT8 model saved: {int8_path}")
    print("\n--- Final INT8 (QAT) accuracy ---")
    qat_acc, qat_f1, qat_report = evaluate(model_int8, val_loader, class_names)
    print(f"QAT-INT8 Accuracy: {qat_acc:.4f} | Macro F1: {qat_f1:.4f}")
    print(qat_report)
    print("\n" + "=" * 62)
    print("SUMMARY -- compare against your PTQ results")
    print("=" * 62)
    print(f"  FP32 baseline      : acc={base_acc:.4f}  f1={base_f1:.4f}")
    print(f"  QAT INT8           : acc={qat_acc:.4f}  f1={qat_f1:.4f}")
    print(f"  QAT vs FP32 delta  : {(qat_acc - base_acc) * 100:+.2f} pp accuracy")
    print("=" * 62)
    with open(os.path.join(args.output_dir, f"{args.backbone}_qat_history.json"), "w") as f:
        json.dump({"history": history, "fp32_baseline": {"accuracy": base_acc, "macro_f1": base_f1}, "qat_int8": {"accuracy": qat_acc, "macro_f1": qat_f1}, "class_weights_used": args.class_weights}, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/mapillary_classification")
    parser.add_argument("--output_dir", type=str, default="models")
    parser.add_argument("--backbone", type=str, default="efficientnet_b0", choices=["baseline_cnn", "mobilenet_v2", "efficientnet_b0"])
    parser.add_argument("--checkpoint", type=str, default="models/efficientnet_b0_best.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--aug_severity", type=str, default="light", choices=["light", "moderate", "heavy"])
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--backend", type=str, default="fbgemm", choices=["fbgemm", "qnnpack"])
    parser.add_argument("--class_weights", action="store_true")
    args = parser.parse_args()
    main(args)


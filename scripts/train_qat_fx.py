import os, sys, argparse, time, json
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.ao.quantization import get_default_qat_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scripts.train_classifier import build_model, AlbumentationsImageFolder
from data.augmentations import get_train_transforms, get_val_transforms

DEVICE = torch.device("cpu")
torch.backends.quantized.engine = "onednn"

def compute_class_weights(dataset):
    labels = [label for _, label in dataset.samples]
    counts = Counter(labels)
    total = sum(counts.values())
    n_classes = len(counts)
    weights = torch.tensor([total / (n_classes * counts[i]) for i in range(n_classes)], dtype=torch.float32)
    print("Class distribution:", dict(sorted(counts.items())))
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
    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    return acc, f1, report

def main(args):
    print(f"FX QAT fine-tuning: {args.backbone}")
    train_ds = AlbumentationsImageFolder(os.path.join(args.data_dir, "train"), get_train_transforms(img_size=args.img_size, severity="light"))
    val_ds = AlbumentationsImageFolder(os.path.join(args.data_dir, "val"), get_val_transforms(img_size=args.img_size))
    class_names = train_ds.classes
    num_classes = len(class_names)
    print(f"Classes: {class_names}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(args.backbone, num_classes)
    ckpt = torch.load(args.checkpoint, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded FP32 checkpoint (val_f1={ckpt.get('val_f1','n/a')})")

    print("\n--- FP32 baseline accuracy ---")
    base_acc, base_f1, _ = evaluate(model, val_loader, class_names)
    print(f"FP32 Accuracy: {base_acc:.4f} | Macro F1: {base_f1:.4f}")

    example_input = torch.randn(1, 3, args.img_size, args.img_size)
    qconfig_mapping = get_default_qat_qconfig_mapping("onednn")

    print("\nPreparing model for FX QAT (auto-traces graph, inserts quant/dequant)...")
    model.train()
    model_prepared = prepare_qat_fx(model, qconfig_mapping, example_input)
    print("Prepared successfully.")

    if args.class_weights:
        weights = compute_class_weights(train_ds)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model_prepared.parameters(), lr=args.lr, weight_decay=1e-4)

    os.makedirs(args.output_dir, exist_ok=True)
    history = []
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
        val_acc, val_f1, _ = evaluate(model_prepared, val_loader, class_names)
        print(f"[Epoch {epoch}/{args.epochs}] train_loss={running_loss/total:.4f} train_acc={correct/total:.4f} | val_acc={val_acc:.4f} val_f1={val_f1:.4f} ({time.time()-t0:.1f}s)")
        history.append({"epoch": epoch, "val_acc": val_acc, "val_f1": val_f1})

    print("\n--- Converting to true INT8 (FX) ---")
    model_prepared.eval()
    model_int8 = convert_fx(model_prepared)
    torch.save(model_int8.state_dict(), os.path.join(args.output_dir, f"{args.backbone}_fxqat_int8.pth"))

    print("\n--- Final QAT-INT8 accuracy ---")
    qat_acc, qat_f1, qat_report = evaluate(model_int8, val_loader, class_names)
    print(f"QAT-INT8 Accuracy: {qat_acc:.4f} | Macro F1: {qat_f1:.4f}")
    print(qat_report)

    print("=" * 60)
    print(f"FP32 baseline : acc={base_acc:.4f}")
    print(f"QAT INT8      : acc={qat_acc:.4f}")
    print(f"Delta         : {(qat_acc-base_acc)*100:+.2f} pp")
    print("Compare against your PTQ result: 0.8862 (88.62%), delta -6.25pp")
    print("=" * 60)

    with open(os.path.join(args.output_dir, f"{args.backbone}_fxqat_history.json"), "w") as f:
        json.dump({"history": history, "fp32": base_acc, "qat_int8": qat_acc}, f, indent=2)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data/mapillary_classification")
    p.add_argument("--output_dir", type=str, default="models")
    p.add_argument("--backbone", type=str, default="efficientnet_b0")
    p.add_argument("--checkpoint", type=str, default="models/efficientnet_b0_best.pt")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--class_weights", action="store_true")
    args = p.parse_args()
    main(args)

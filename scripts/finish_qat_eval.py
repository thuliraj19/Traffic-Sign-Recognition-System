import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import torch
torch.backends.quantized.engine = "onednn"

from torch.ao.quantization import get_default_qat_qconfig, prepare_qat, convert
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm

from scripts.train_classifier import build_model, AlbumentationsImageFolder
from data.augmentations import get_val_transforms

DEVICE = torch.device("cpu")

ckpt = torch.load("models/efficientnet_b0_qat.pt", map_location=DEVICE)
model = build_model(ckpt["backbone"], ckpt["num_classes"])
model.qconfig = get_default_qat_qconfig("onednn")
model_prepared = prepare_qat(model, inplace=False)
model_prepared.load_state_dict(ckpt["model_state_dict"])
model_prepared.eval()

print("Converting to INT8...")
model_int8 = convert(model_prepared, inplace=False)
torch.save(model_int8.state_dict(), "models/efficientnet_b0_qat_int8.pth")
print("Saved: models/efficientnet_b0_qat_int8.pth")

val_ds = AlbumentationsImageFolder("data/mapillary_classification/val", get_val_transforms(img_size=ckpt["img_size"]))
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
class_names = val_ds.classes

all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in tqdm(val_loader, desc="Evaluating QAT-INT8"):
        outputs = model_int8(images)
        all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.numpy())

acc = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average="macro")
print(f"\nQAT-INT8 Accuracy: {acc:.4f}")
print(f"QAT-INT8 Macro F1: {f1:.4f}")
print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

print("=" * 60)
print(f"Saved val_f1 during training was: {ckpt.get('val_f1', 'n/a')}")
print("Compare this QAT-INT8 accuracy against your earlier PTQ result: 0.8862 (88.62%)")
print("=" * 60)




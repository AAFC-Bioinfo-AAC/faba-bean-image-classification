import argparse
import os, glob, random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image, ImageOps
import segmentation_models_pytorch as smp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

# ---------------------------
# Dataset
# ---------------------------
class CoinDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, augment=False):
        self.transform = transform
        self.augment = augment
        self.pairs = []
        for img_path in glob.glob(os.path.join(image_dir, "*.JPG")):
            base = os.path.splitext(os.path.basename(img_path))[0]
            mask_path = os.path.join(mask_dir, f"{base}_coin.png")
            if os.path.exists(mask_path):
                self.pairs.append((img_path, mask_path))
            else:
                print(f"Warning: Mask not found for {img_path}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        image = ImageOps.exif_transpose(Image.open(img_path).convert("RGB"))
        mask  = ImageOps.exif_transpose(Image.open(mask_path).convert("L"))

        # Augmentation
        if self.augment:
            if random.random() > 0.5:
                image = transforms.functional.hflip(image)
                mask = transforms.functional.hflip(mask)
            if random.random() > 0.5:
                image = transforms.functional.vflip(image)
                mask = transforms.functional.vflip(mask)
            angle = random.choice([0, 90, 180, 270])
            if angle != 0:
                image = transforms.functional.rotate(image, angle)
                mask = transforms.functional.rotate(mask, angle)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        mask = (mask > 0).float()
        return image, mask

# ---------------------------
# Metrics
# ---------------------------
def segmentation_metrics(preds, gts, threshold=0.5):
    preds = (preds > threshold).float()
    gts   = (gts > 0.5).float()
    TP = (preds * gts).sum()
    FP = (preds * (1 - gts)).sum()
    FN = ((1 - preds) * gts).sum()
    TN = ((1 - preds) * (1 - gts)).sum()
    eps = 1e-6
    return {
        "IoU": (TP / (TP + FP + FN + eps)).item(),
        "Dice": (2 * TP / (2 * TP + FP + FN + eps)).item(),
        "Pixel_Accuracy": ((TP + TN) / (TP + TN + FP + FN + eps)).item(),
        "Precision": (TP / (TP + FP + eps)).item(),
        "Recall": (TP / (TP + FN + eps)).item(),
        "Specificity": (TN / (TN + FP + eps)).item(),
        "FPR": (FP / (FP + TN + eps)).item(),
        "FNR": (FN / (FN + TP + eps)).item(),
    }

# ---------------------------
# Confusion Matrix
# ---------------------------
def compute_confusion_matrix(model, loader, device, threshold=0.5):
    model.eval()
    preds_all, gts_all = [], []
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            outputs = model(imgs)
            preds = (outputs > threshold).float()
            preds_all.append(preds.cpu().view(-1))
            gts_all.append(masks.cpu().view(-1))
    preds_all = torch.cat(preds_all).numpy()
    gts_all   = torch.cat(gts_all).numpy()
    return confusion_matrix(gts_all, preds_all)

def plot_confusion_matrix(cm, title, out_path, normalize=True):
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(5,4))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=["Background", "Coin"],
        yticklabels=["Background", "Coin"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

# ---------------------------
# Plot metrics
# ---------------------------
def plot_training_curves(df, out_dir="graphs", prefix="coin"):
    os.makedirs(out_dir, exist_ok=True)
    for col in df.columns:
        if col == "Epoch":
            continue
        plt.figure()
        plt.plot(df["Epoch"], df[col], marker="o")
        plt.xlabel("Epoch")
        plt.ylabel(col.replace("_", " "))
        plt.title(col.replace("_", " ") + " vs Epoch")
        plt.grid(True)
        filename = f"{prefix}_{col}.png"
        plt.savefig(os.path.join(out_dir, filename), dpi=300)
        plt.close()

# ---------------------------
# Main Training
# ---------------------------
def main(args):
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor()
    ])
    dataset = CoinDataset(args.train_images, args.train_masks, transform, augment=True)
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = smp.Unet(
        encoder_name="mit_b0",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation="sigmoid"
    ).to(device)

    dice_loss = smp.losses.DiceLoss(mode="binary")
    bce_loss  = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    history = []

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = dice_loss(preds, masks) + bce_loss(preds, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        metric_sum = {k: 0 for k in segmentation_metrics(torch.zeros(1), torch.zeros(1)).keys()}
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs)
                loss = dice_loss(preds, masks) + bce_loss(preds, masks)
                val_loss += loss.item() * imgs.size(0)
                batch_metrics = segmentation_metrics(preds, masks)
                for k in metric_sum:
                    metric_sum[k] += batch_metrics[k] * imgs.size(0)
        val_loss /= len(val_loader.dataset)
        metrics_avg = {k: v / len(val_loader.dataset) for k, v in metric_sum.items()}

        history.append({
            "Epoch": epoch + 1,
            "Train_Loss": train_loss,
            "Val_Loss": val_loss,
            **metrics_avg
        })

        print(f"Epoch {epoch+1:02d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Dice: {metrics_avg['Dice']:.4f}")

    # ---------------------------
    # Save metrics + plots
    # ---------------------------
    prefix = "coin"
    df = pd.DataFrame(history)
    metrics_file = f"{prefix}_training_metrics.csv"
    df.to_csv(metrics_file, index=False)
    plot_training_curves(df, out_dir="graphs", prefix=prefix)

    # Confusion matrices
    cm_train = compute_confusion_matrix(model, train_loader, device)
    cm_val   = compute_confusion_matrix(model, val_loader, device)
    plot_confusion_matrix(cm_train, "Train Confusion Matrix", f"{prefix}_cm_train.png")
    plot_confusion_matrix(cm_val, "Validation Confusion Matrix", f"{prefix}_cm_val.png")

    # Save model
    model_file = f"{prefix}_unet_mitb0_coin.pth"
    torch.save(model.state_dict(), model_file)

    print("\n✅ Training complete")
    print("📊 Metrics table saved:", metrics_file)
    print("📈 Metric plots saved in: graphs/")
    print("🧮 Confusion matrices saved:", f"{prefix}_cm_train.png", f"{prefix}_cm_val.png")
    print("💾 Model saved:", model_file)

# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UNet-MiT on Single-Coin Images")
    parser.add_argument("--train_images", required=True)
    parser.add_argument("--train_masks", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--val_split", type=float, default=0.1)
    args = parser.parse_args()
    main(args)

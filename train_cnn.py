main training code "# train_cnn.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
import os, json

# -----------------------------
# Config
# -----------------------------
DATA_DIR = "C:\Users\Admin\OneDrive\Desktop\ML/dataset_binary"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "plastic_cnn.pth")
META_PATH = os.path.join(MODEL_DIR, "meta_classes.json")

BATCH_SIZE = 16
NUM_EPOCHS = 30  # train longer, but stop early if needed
PATIENCE = 4     # early stopping patience
LR = 1e-4

os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# Data transforms
# -----------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

# -----------------------------
# Dataset & Dataloader
# -----------------------------
train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)
val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

print("Classes (train):", train_ds.classes)
class_to_idx = train_ds.class_to_idx
with open(META_PATH, "w") as f:
    json.dump({"classes": train_ds.classes, "class_to_idx": class_to_idx}, f)

# -----------------------------
# Model
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(train_ds.classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# -----------------------------
# Training loop with Early Stopping
# -----------------------------
best_val_acc = 0.0
patience_counter = 0

for epoch in range(1, NUM_EPOCHS+1):
    # ---- Train ----
    model.train()
    total_loss, total_correct = 0, 0
    for imgs, labels in tqdm(train_loader, desc=f"Train Epoch {epoch}/{NUM_EPOCHS}"):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        total_correct += (outputs.argmax(1) == labels).sum().item()

    train_loss = total_loss / len(train_ds)
    train_acc = total_correct / len(train_ds)

    # ---- Validation ----
    model.eval()
    val_loss, val_correct = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * imgs.size(0)
            val_correct += (outputs.argmax(1) == labels).sum().item()

    val_loss /= len(val_ds)
    val_acc = val_correct / len(val_ds)

    scheduler.step()

    print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | Val Loss={val_loss:.4f}, Acc={val_acc:.4f}")

    # ---- Early Stopping ----
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"✅ Best model saved at epoch {epoch} (Val Acc={val_acc:.4f})")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("⏹ Early stopping triggered!")
            break
"H

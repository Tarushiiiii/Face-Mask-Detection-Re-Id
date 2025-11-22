import torch
from torch.utils.data import DataLoader
from torch import optim, nn
from tqdm import tqdm   # <-- NEW

from model import ReIDModel
from dataset import Market1501
from trainer import Trainer
from utils import get_transform, save_checkpoint
from config import DATA_DIR, BATCH_SIZE, LR, EPOCHS, MODEL_SAVE_PATH, IMG_SIZE


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("\nLoading Market1501 dataset... please wait.\n")

    # -----------------------------------------------
    # Load Dataset
    # -----------------------------------------------
    transform = get_transform(IMG_SIZE)
    train_dataset = Market1501(DATA_DIR, transform=transform, mode="train")

    print(f"Dataset Loaded: {len(train_dataset)} training images\n")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # -----------------------------------------------
    # Load Model
    # -----------------------------------------------
    print("Initializing model...")
    model = ReIDModel(num_classes=train_dataset.num_classes)
    model = model.to(device)
    print("✔ Model ready.\n")

    # -----------------------------------------------
    # Define Loss + Optimizer
    # -----------------------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    trainer = Trainer(
        model=model,
        dataloader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device
    )

    # -----------------------------------------------
    # Training Loop
    # -----------------------------------------------
    print("Starting training...\n")

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        # tqdm progress bar for batches
        epoch_loss = 0
        for batch_loss in tqdm(trainer.train_one_epoch(return_batches=True), desc="Training", unit="batch"):
            epoch_loss += batch_loss

        avg_loss = epoch_loss / len(train_loader)

        print(f"Epoch {epoch+1} Completed — Loss: {avg_loss:.4f}")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_checkpoint(model, f"{MODEL_SAVE_PATH}_epoch{epoch+1}.pth")
            print(f"Checkpoint saved at epoch {epoch+1}")

    save_checkpoint(model, MODEL_SAVE_PATH)
    print("\nTraining finished! Model saved at:", MODEL_SAVE_PATH)


if __name__ == "__main__":
    main()

import torch
from torch.utils.data import DataLoader
from torch import optim, nn

from model import ReIDModel
from dataset import Market1501
from trainer import Trainer
from utils import get_transform, save_checkpoint
from config import DATA_DIR, BATCH_SIZE, LR, EPOCHS, MODEL_SAVE_PATH, IMG_SIZE


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -----------------------------------------------
    # Load Dataset
    # -----------------------------------------------
    transform = get_transform(IMG_SIZE)
    train_dataset = Market1501(DATA_DIR, transform=transform, mode="train")

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
    model = ReIDModel(num_classes=train_dataset.num_classes)
    model = model.to(device)

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
    for epoch in range(EPOCHS):
        loss = trainer.train_one_epoch()
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss:.4f}")

        # Save checkpoint every 5 epochs (optional)
        if (epoch + 1) % 5 == 0:
            save_checkpoint(model, f"{MODEL_SAVE_PATH}_epoch{epoch+1}.pth")

    # Final model save
    save_checkpoint(model, MODEL_SAVE_PATH)
    print("Training finished! Model saved at", MODEL_SAVE_PATH)


if __name__ == "__main__":
    main()

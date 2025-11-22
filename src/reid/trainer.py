import torch

class Trainer:
    def __init__(self, model, dataloader, optimizer, criterion, device):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_one_epoch(self, return_batches=False):
        self.model.train()
        total_loss = 0.0
        n = 0

        for imgs, labels, _cams in self.dataloader:
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)

            embed, logits = self.model(imgs)
            loss = self.criterion(logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)

            if return_batches:
                yield loss.item()   # batch-level loss for tqdm

        if not return_batches:
            return total_loss / n if n > 0 else 0.0


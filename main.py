from dataloader import get_fashion_loaders
from model import TransformedIlluminator, RecurrentIlluminator
from trainer import Trainer
from torch.optim import Adam
import wandb


def main():
    wandb.init('adaptive_imager')
    train_loader, val_loader, test_loader, channels, classes = get_fashion_loaders(batch_size=64)
    model = TransformedIlluminator(num_leds=channels, num_classes=classes)
    # model = RecurrentIlluminator(num_leds=channels, num_classes=classes)
    model.cuda()
    optimizer = Adam(model.parameters())
    trainer = Trainer(model=model, optimizer=optimizer, num_epochs=25, train_loader=train_loader, val_loader=val_loader,
                      test_loader=test_loader, max_trajectory_length=3)

    trainer.train()


if __name__ == "__main__":
    main()

from dataloader import get_fashion_loaders, get_mnist_loaders, get_malaria_loaders
from model import TransformedIlluminator, RecurrentIlluminator
from trainer import Trainer
from torch.optim import Adam
import wandb


def main():
    wandb.init('adaptive_imager')
    train_loader, val_loader, test_loader, channels, classes = get_mnist_loaders(batch_size=128)
    model = TransformedIlluminator(num_leds=channels, num_classes=classes, std=0.0)
    # model = RecurrentIlluminator(num_leds=channels, num_classes=classes, std=0.0)
    #model.cuda()

    optimizer = Adam(model.parameters())
    trainer = Trainer(model=model, optimizer=optimizer, num_epochs=100, train_loader=train_loader, val_loader=val_loader,
                      test_loader=test_loader, max_trajectory_length=2)
    trainer.train()


if __name__ == "__main__":
    main()

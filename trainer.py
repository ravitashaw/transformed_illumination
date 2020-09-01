from utils import AverageMeter
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import wandb


class Trainer:

    def __init__(self, model, optimizer, num_epochs, train_loader, val_loader, test_loader=None,
                 max_trajectory_length=8):
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.train_loader = train_loader
        self.num_epochs = num_epochs
        self.model = model
        self.optimizer = optimizer
        self.max_trajectory_length = max_trajectory_length
        self.classifier_loss = nn.CrossEntropyLoss()
        self.decision_loss = nn.CrossEntropyLoss(reduction='none')
        self.use_gpu = self.model.device == 'cuda'

    def train(self):
        for self.curr_epoch in range(self.num_epochs):
            # TODO: add num_glimpses
            train_loss, train_acc, sample_images = self.run_one_epoch(self.train_loader, True)
            val_loss, val_acc, sample_images = self.run_one_epoch(self.val_loader, False)
            metrics = {
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            }
            if self.test_loader is not None:
                test_loss, test_acc = self.run_one_epoch(self.test_loader, False)
                metrics.update({
                    'test_loss': test_loss,
                    'test_acc': test_acc
                })
            # TODO: add wandb support
            print(self.curr_epoch, metrics)
            illuminated_images = [wandb.Image(sample_image, caption=f"Shot {j}") for j, sample_image in
                                  enumerate(sample_images)]
            wandb.log({"illumination_traj": illuminated_images}, step=self.curr_epoch)

    def run_one_epoch(self, loader, training):
        losses = AverageMeter()
        accs = AverageMeter()
        if training:
            self.model.train()
        else:
            self.model.eval()
        with tqdm(total=len(loader)) as pbar:
            for i, data in enumerate(loader):
                x, y = data
                if self.use_gpu:
                    x, y = x.cuda(), y.cuda()
                if training:
                    self.optimizer.zero_grad()
                    loss, acc, sample_images = self.rollout(x, y)
                    loss.backward()
                    self.optimizer.step()
                else:
                    with torch.no_grad():
                        loss, acc, sample_images = self.rollout(x, y)
                loss_data = float(loss.data)
                acc_data = float(acc)
                losses.update(loss_data)
                accs.update(acc_data)
                pbar.update(1)
                pbar.set_description(desc=f"loss: {losses.avg:.3f} acc: {accs.avg:.3f}")
        return losses.avg, accs.avg, sample_images

    def rollout(self, x_data, y_data):
        phi = None
        zs = None
        batch_size = x_data.shape[0]
        final_classifications = [None] * batch_size
        final_decisions = [None] * batch_size
        early_exit = False
        trajectories = [-1] * batch_size
        timeouts = [False] * batch_size
        glimpses = [0] * batch_size
        sample_images = []
        # do rollout
        for i in range(self.max_trajectory_length):
            decisions, classifications, phi, zs, image = self.model(x_data, phi, zs)
            sample_images.append(image[0, 0].detach().cpu().numpy())
            for b in range(batch_size):
                if final_decisions[b] is not None:
                    continue
                else:
                    if torch.argmax(decisions[b]) == 1:
                        final_decisions[b] = decisions[b]
                        # final_classifications[b] = classifications[b]
                        trajectories[b] = i
            # if all([decision is not None for decision in final_decisions]):
            #     early_exit = True
            #     break
        if not early_exit:
            for j in range(batch_size):
                if final_classifications[j] is None:
                    final_decisions[j] = decisions[j]
                    final_classifications[j] = classifications[j]
                    timeouts[j] = True
                    trajectories[j] = self.max_trajectory_length
        final_classifications = torch.stack(final_classifications, dim=0)
        final_decisions = torch.stack(final_decisions, dim=0)
        classification_loss = self.classifier_loss(final_classifications, y_data)
        # for the decision lets start with something simple, as in a "wrong classification" penalty and a "timeout"
        # penalty
        correct_classification = (torch.argmax(final_classifications, dim=-1) == y_data).float()
        decision_target = []
        decision_scaling = []
        for i in range(batch_size):
            if timeouts[i]:
                decision_target.append(1)
                decision_scaling.append(1.0)
            elif correct_classification[i] == 1:
                decision_target.append(1)
                decision_scaling.append(1.0)
            elif correct_classification[i] == 0:
                decision_target.append(0)
                decision_scaling.append(1.0)
            else:
                raise RuntimeError()
        decision_target = torch.tensor(decision_target, device=final_classifications.device)
        decision_scaling = torch.tensor(decision_scaling, device=final_classifications.device)
        decision_loss = (self.decision_loss(final_decisions, decision_target) * decision_scaling).mean()
        # total_loss = classification_loss + decision_loss
        total_loss = classification_loss
        acc = correct_classification.sum() / len(y_data)

        return total_loss, acc, sample_images

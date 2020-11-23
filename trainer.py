from utils import AverageMeter
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt


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
        self.use_gpu = False  # self.model.device == 'cuda'

    @staticmethod
    def make_wandb_images(image_sequence, title=""):
        normalized = plt.Normalize()
        norm = lambda im: (im - np.min(im)) / (np.max(im) - np.min(im))
        return [wandb.Image(plt.cm.viridis(normalized(norm(sample_image))),
                            caption=f"Shot {j+1} ({title}, mean={sample_image.mean()})") for j, sample_image in
                enumerate(image_sequence)]

    def train(self, adaptive_trajectory):
        for self.curr_epoch in range(self.num_epochs):
            train_loss, train_acc, train_traj = self.run_one_epoch(self.train_loader, True, adaptive_trajectory)
            val_loss, val_acc, val_traj = self.run_one_epoch(self.val_loader, False, adaptive_trajectory)
            metrics = {
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'train_traj': train_traj,
                'val_traj': val_traj
            }
            if self.test_loader is not None:
                test_loss, test_acc = self.run_one_epoch(self.test_loader, False, adaptive_trajectory)
                metrics.update({
                    'test_loss': test_loss,
                    'test_acc': test_acc
                })
            wandb.log(metrics, step=self.curr_epoch)

    def run_one_epoch(self, loader, training, adaptive_trajectory):
        losses = AverageMeter()
        accs = AverageMeter()
        trajs = AverageMeter()

        keyword = 'Training' if training else 'Validation'
        self.model.train() if training else self.model.eval()

        with tqdm(total=len(loader)) as pbar:
            for i, data in enumerate(loader):
                x, y = data
                if self.use_gpu:
                    x, y = x.cuda(), y.cuda()
                if training:
                    self.optimizer.zero_grad()
                    loss, acc, sample_images, sample_phi, traj, sample_traj_length = self.rollout(x, y, adaptive_trajectory)
                    loss.backward()
                    self.optimizer.step()
                else:
                    with torch.no_grad():
                        loss, acc, sample_images, sample_phi, traj, sample_traj_length = self.rollout(x, y, adaptive_trajectory)
                loss_data = float(loss.data)
                acc_data = float(acc)
                traj_data = float(traj)
                losses.update(loss_data)
                accs.update(acc_data)
                trajs.update(traj_data)
                pbar.update(1)
                pbar.set_description(desc=f"Epoch {self.curr_epoch} {keyword} Loss: {losses.avg:.3f} Accuracy: {accs.avg:.3f} Avg Traj: {trajs.avg:.3f}")

                title = f'Traj Length: {sample_traj_length}'
                illuminated_images = self.make_wandb_images(sample_images, title)
                sample_phi_ph = self.make_wandb_images(sample_phi, title)
                wandb.log({f"{keyword}_Images": illuminated_images}, step=self.curr_epoch)
                wandb.log({f"{keyword}_Illunmination": sample_phi_ph}, step=self.curr_epoch)

        return losses.avg, accs.avg, trajs.avg

    def rollout(self, x_data, y_data, adaptive_trajectory):

        batch_size = x_data.shape[0]
        phi = None  # led pattern Updated for each image
        zs = None  # next hidden state/embedding
        sample_images = []
        sample_phi = []
        first_traj_length = -1

        final_classifications = [None] * batch_size  # Final Classification Probs between the classes
        final_decisions = [None] * batch_size  # Decision Probs if can classify or require phi
        done_indices = [-1] * batch_size  # Number of Glimpses
        timeouts = [False] * batch_size  # All Glimpses are finished and still no decision to classify
        glimpses = [-1] * batch_size  # Number of glimpses for each image to decision and classify - trajectory length

        # iterate through each of the trajectory
        for t in range(self.max_trajectory_length):
            decisions, classifications, phi, zs, image = self.model(x_data, phi, zs)

            # for sampling first image of batch in wandb
            sample_images.append(image[0, 0].detach().cpu().numpy())
            sample_phi.append(phi[0, :].detach().cpu().numpy())

            # for each image in the batch
            for b_idx in range(batch_size):

                # check if glimpse already classified
                if done_indices[b_idx] > -1:
                    continue;
                # else if not done, check if decision is good for classification now
                elif torch.argmax(decisions[b_idx]) == 1:
                    # update the tracker with current glimpse
                    done_indices[b_idx] = 1

                    if glimpses[b_idx] == -1:
                        # mark it as done
                        if b_idx == 0:
                            first_traj_length = t+1  # for wandb
                        glimpses[b_idx] = t + 1
                    # save Final Classification Probability
                    final_classifications[b_idx] = classifications[b_idx]
                    # save Final Decision Probability
                    final_decisions[b_idx] = decisions[b_idx]
                # else if decision is not good and the glimpses are timing out
                elif t == self.max_trajectory_length - 1:
                    # update the current time out
                    timeouts[b_idx] = True
                    # update the tracker with current glimpse
                    glimpses[b_idx] = t + 1
                    # mark it as done
                    done_indices[b_idx] = 0
                    # save Final Classification Probability
                    final_classifications[b_idx] = classifications[b_idx]
                    # save Final Decision Probability
                    final_decisions[b_idx] = decisions[b_idx]
                # else go to next trajectory and make new decision (next iteration)

            # if all the images are classified before the max trajectory length has reached
            # break the trajectory loop, this is the adaptive trajectory part
            if all([done_index > -1 for done_index in done_indices]) and adaptive_trajectory:
                break

        # Classification Loss
        final_classifications = torch.stack(final_classifications).float()
        classification_loss = self.classifier_loss(final_classifications, y_data)
        correct_classification = (torch.argmax(final_classifications, dim=-1) == y_data).float()

        # Decision Loss
        final_decisions = torch.stack(final_decisions)
        decision_target = []  # target for decision based upon timeout and correct classification
        decision_scaling = []  # penalty scale for each case

        # for each image
        for i in range(batch_size):
            # if timed out, no decision was made, no classification happened (stay decision reward)
            # Punish number of glimpses time, so optimization occurs with less number of glimpses
            if timeouts[i]:
                decision_target.append(1)
                decision_scaling.append(1.0)
            # if classification was correct (exit decision penalty)
            elif correct_classification[i] == 1:
                decision_target.append(1)  # reward
                decision_scaling.append(0.01)
            # if incorrect classification, penalize (wrong decision is same as stay decision reward)
            elif correct_classification[i] == 0:
                decision_target.append(0)
                decision_scaling.append(1.0)
            else:
                raise RuntimeError()

        decision_target = torch.tensor(decision_target)
        decision_scaling = torch.tensor(decision_scaling)
        decision_loss = (self.decision_loss(final_decisions, decision_target) * decision_scaling).mean()

        loss = classification_loss + decision_loss
        acc = correct_classification.sum() / len(y_data)

        # average trajectory
        trajs = torch.tensor(glimpses).float().mean()

        return loss, acc, sample_images, sample_phi, trajs, first_traj_length

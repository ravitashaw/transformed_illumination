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
        self.use_gpu = False #self.model.device == 'cuda'

    @staticmethod
    def make_wandb_images(image_sequence):
        normalized = plt.Normalize()
        norm = lambda im: (im - np.min(im)) / (np.max(im) - np.min(im))
        return [wandb.Image(plt.cm.viridis(normalized(norm(sample_image))), caption=f"Shot {j}") for j, sample_image in
                enumerate(image_sequence)]

    def train(self):
        for self.curr_epoch in range(self.num_epochs):
            # TODO: add num_glimpses
            train_loss, train_acc, sample_images_train, sample_phi_train, train_traj  = self.run_one_epoch(self.train_loader, True)
            val_loss, val_acc, sample_images_val, sample_phi_val, val_traj = self.run_one_epoch(self.val_loader, False)
            metrics = {
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'train_traj': train_traj,
                'val_traj': val_traj
            }
            if self.test_loader is not None:
                test_loss, test_acc = self.run_one_epoch(self.test_loader, False)
                metrics.update({
                    'test_loss': test_loss,
                    'test_acc': test_acc
                })
            # TODO: add wandb support
            illuminated_images_train = self.make_wandb_images(sample_images_train)
            illuminated_images_val = self.make_wandb_images(sample_images_val)
            sample_phi_train_ph = self.make_wandb_images(sample_phi_train)
            sample_phi_val_ph = self.make_wandb_images(sample_phi_val)
            wandb.log({"illumination_traj_train": illuminated_images_train}, step=self.curr_epoch)
            wandb.log({"illumination_traj_val": illuminated_images_val}, step=self.curr_epoch)
            wandb.log({"sample_phi_train_ph": sample_phi_train_ph}, step=self.curr_epoch)
            wandb.log({"sample_phi_val_ph": sample_phi_val_ph}, step=self.curr_epoch)
            wandb.log(metrics, step=self.curr_epoch)

    def run_one_epoch(self, loader, training):
        losses = AverageMeter()
        accs = AverageMeter()
        trajs = AverageMeter()

        ss = None
        keyword = 'Training'
        if training:
            self.model.train()
        else:
            keyword = 'Validation'
            self.model.eval()
        with tqdm(total=len(loader)) as pbar:
            for i, data in enumerate(loader):
                x, y = data
                if self.use_gpu:
                    x, y = x.cuda(), y.cuda()
                if training:
                    self.optimizer.zero_grad()
                    loss, acc, sample_images, sample_phi, traj = self.rollout(x, y, adaptive_trajectory=False)
                    loss.backward()
                    self.optimizer.step()
                else:
                    with torch.no_grad():
                        loss, acc, sample_images, sample_phi, traj = self.rollout(x, y, adaptive_trajectory=False)
                loss_data = float(loss.data)
                acc_data = float(acc)
                traj_data = float(traj)
                losses.update(loss_data)
                accs.update(acc_data)
                trajs.update(traj_data)
                pbar.update(1)
                pbar.set_description(desc=f"{keyword} loss: {losses.avg:.3f} acc: {accs.avg:.3f} traj: {trajs.avg:.3f}")
                if ss is None:
                    ss = sample_images
        return losses.avg, accs.avg, ss, sample_phi, trajs.avg

    def rollout(self, x_data, y_data, adaptive_trajectory=False):

        batch_size = x_data.shape[0]
        phi = None  # led pattern Updated for each image
        zs = None  # next hidden state/embedding
        glimpse_number = 0
        sample_images = []
        sample_phi = []

        final_classifications = [None] * batch_size  # Final Classification Probs between the classes
        final_decisions = [None] * batch_size  # Decision Probs if can classify or require phi
        done_indices = [-1] * batch_size  # Number of Glimpses
        timeouts = [False] * batch_size  # All Glimpses are finished and still no decision to classify
        glimpses = [0] * batch_size  # Number of glimpses for each image to decision and classify - trajectory length
        log_pis = []

        # iterate through each of the trajectory
        for t in range(self.max_trajectory_length):
            glimpse_number += 1
            decisions, classifications, phi, zs, image, p = self.model(x_data, phi, zs)
            log_pis.append(p)

            # for each image in the batch
            for b_idx in range(batch_size):
                # check if glimpse already classified
                if done_indices[b_idx] > -1:
                    continue;
                # else if not done, check if decision is good for classification now
                elif torch.argmax(decisions[b_idx]) == 1:
                    # update the tracker with current glimpse
                    glimpses[b_idx] = t + 1
                    # mark it as done
                    done_indices[b_idx] = 1
                    # save Final Classification Probability
                    final_classifications[b_idx] = classifications[b_idx]
                    # save Final Decision Probability
                    final_decisions[b_idx] = decisions[b_idx]
                # else if decision is not good and the glimpses are timing out
                elif t == self.max_trajectory_length-1:
                    # update the current time out
                    timeouts[b_idx] = True
                    # update the tracker with current glimpse
                    glimpses[b_idx] = t + 1
                    # mark it as done
                    done_indices[b_idx] = 1
                    # save Final Classification Probability
                    final_classifications[b_idx] = classifications[b_idx]
                    # save Final Decision Probability
                    final_decisions[b_idx] = decisions[b_idx]
                # else go to next trajectory and make new decision (next iteration)
                if b_idx == 0:
                    sample_images.append(image[0, 0].detach().cpu().numpy())
                    sample_phi.append(phi[0, :].detach().cpu().numpy())

            # if all the images are classified before the max trajectory length has reached
            # break the trajectory loop, this is the adaptive trajectory part
            if all([done_index > -1 for done_index in done_indices]) and adaptive_trajectory:
                break

        # Classification Loss
        final_classifications = torch.stack(final_classifications)
        classification_loss = self.classifier_loss(final_classifications, y_data)
        correct_classification = (torch.argmax(final_classifications, dim=-1) == y_data).float()

        # Decision Loss
        final_decisions = torch.stack(final_decisions)
        decision_target = []  # target for decision based upon timeout and correct classification
        decision_scaling = [] # penalty scale for each case

        # for each image
        for i in range(batch_size):
            # if timed out, no decision was made, no classification happened (stay decision reward)
            if timeouts[i]:
                decision_target.append(1)
                decision_scaling.append(1.0 * glimpses[b_idx])
            # if classification was correct (exit decision reward)
            elif correct_classification[i] == 1:
                decision_target.append(1)
                decision_scaling.append(0.01)
            # if incorrect classification, penalize (wrong decision is same as stay decision reward)
            elif correct_classification[i] == 0:
                decision_target.append(0)
                decision_scaling.append(1.0 * glimpses[b_idx])
            else:
                raise RuntimeError()

        decision_target = torch.tensor(decision_target)
        decision_scaling = torch.tensor(decision_scaling)
        decision_loss = (self.decision_loss(final_decisions, decision_target) * decision_scaling).mean()

        # average trajectory
        trajs = torch.tensor(glimpses).float().mean()

        loss = classification_loss + decision_loss
        acc = correct_classification.sum() / len(y_data)

        return loss, acc, sample_images, sample_phi, trajs






























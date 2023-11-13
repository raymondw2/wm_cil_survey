import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from models.podnet import pod_spatial_loss
from utils.inc_net import IncrementalNet
from utils.toolkit import target2onehot, tensor2numpy

import sys
from watermaze.environment import *

def one_hot(num):
    """
    Given a number between 0-3, returns a one-hot tensor in PyTorch.
    """
    one_hot_vec = torch.zeros(4)
    one_hot_vec[num] = 1
    return one_hot_vec
EPSILON = 1e-8

init_epoch = 1
init_lr = 0.001
init_milestones = [60, 120, 170]
init_lr_decay = 0.1
init_weight_decay = 0.0005


epochs = 1
lrate = 0.001
milestones = [70, 120, 150]
lrate_decay = 0.1
batch_size = 1
weight_decay = 2e-4
num_workers = 4
T = 2
lamda = 1000
fishermax = 0.0001
SEQUENCE_LENGTH = 100

class EWC(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.fisher = None
        self._network = IncrementalNet(args["convnet_type"], False)
        
        NUM_OF_ENVS = 5

        self.envs = {}
        for i in np.arange(NUM_OF_ENVS):
            env_name = f"{i}" #to keep 1 index names
            self.envs[env_name] = SquareMaze(size = 15, observation_size = 1000, name = f"{i}")
            

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1 
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        #TODO Check
        self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        if self.fisher is None:
            self.fisher = self.getFisherDiagonal(self.train_loader)
        else:
            alpha = self._known_classes / self._total_classes
            new_finsher = self.getFisherDiagonal(self.train_loader)
            for n, p in new_finsher.items():
                new_finsher[n][: len(self.fisher[n])] = (
                    alpha * self.fisher[n]
                    + (1 - alpha) * new_finsher[n][: len(self.fisher[n])]
                )
            self.fisher = new_finsher
        self.mean = {
            n: p.clone().detach()
            for n, p in self._network.named_parameters()
            if p.requires_grad
        }

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._cur_task == 0:
            optimizer = optim.Adam(
                self._network.parameters(),
                lr=init_lr,
                )
            if False:
                optimizer = optim.SGD(
                    self._network.parameters(),
                    lr=init_lr,
                    momentum=0.9,
                    weight_decay=init_weight_decay,
                )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=init_milestones, gamma=init_lr_decay
            )
            
            if self.args['skip']:
                if len(self._multiple_gpus) > 1:
                    self._network = self._network.module
                load_acc = self._network.load_checkpoint(self.args)
                self._network.to(self._device)
                cur_test_acc = self._compute_accuracy(self._network, self.test_loader)
                logging.info(f"Loaded_Test_Acc:{load_acc} Cur_Test_Acc:{cur_test_acc}")
                if len(self._multiple_gpus) > 1:
                    self._network = nn.DataParallel(self._network, self._multiple_gpus)
            else:
                self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            optimizer = optim.Adam(
                self._network.parameters(),
                lr=lrate,
                )
            if False:
                optimizer = optim.SGD(
                    self._network.parameters(),
                    lr=lrate,
                    momentum=0.9,
                    weight_decay=weight_decay,
                )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=milestones, gamma=lrate_decay
            )
            self._update_representation(train_loader, test_loader, optimizer, scheduler)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
      
        #breakpoint()
        prog_bar = tqdm(range(init_epoch))
        for _, epoch in enumerate(prog_bar):

            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            e_t = tqdm(enumerate(train_loader))
            for i, (_, inputs, targets) in e_t:
                # TODO Given the start and goal location, run exp.

                env = self.envs[str(targets.item())]
                #breakpoint()

                env.reset(inputs[0][:2].cpu().numpy(), inputs[0][2].item())

                loss = 0
                #breakpoint()
                count = 0
                for _ in range(SEQUENCE_LENGTH):
                    if env.success:
                        break
                    obs = torch.Tensor(env.get_vision()).to(self._device).reshape(1,1000)
                    head_direction = one_hot(env.prev_move_global).to(self._device)

                    pred = self._network([obs, head_direction])
                    pred = pred["logits"]
                    y = torch.tensor(env.find_optimal_move()).to(self._device).long()
                    _loss = F.cross_entropy(pred, y) #TODO add loss function
                    loss += _loss
                    count += 1

                    pred_step = torch.argmax(pred)
                    #breakpoint()
                    env.step(pred_step)
                
                loss /= count
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                correct += int(env.success)
                e_t.set_postfix(correct = correct)
                total += 1
                #inputs, targets = inputs.to(self._device), targets.to(self._device)
                #logits = self._network(inputs)["logits"]
                #loss = F.cross_entropy(logits, targets)
                #optimizer.zero_grad()
                #loss.backward()
                #optimizer.step()
                #losses += loss.item()

                #_, preds = torch.max(logits, dim=1)
                #correct += preds.eq(targets.expand_as()).cpu().sum()
                #total += len(targets)

            scheduler.step()
            train_acc = np.around((correct) * 100 / total, decimals=2)

            if epoch % 5 == 0:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
                    train_acc,
                )
            else:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            prog_bar.set_description(info)

        logging.info(info)

    def eval_task(self, save_conf = False):
        test_loader = self.test_loader
        #TODO
        
        self._network.eval()
        losses = 0.0
        correct, total = 0, 0
        grouped = {i: [0, 0] for i in range(5)}
        e_t = tqdm(enumerate(test_loader))
        for i, (_, inputs, targets) in e_t:
            # TODO Given the start and goal location, run exp.
            env = self.envs[str(targets.item())]
            #breakpoint()
            env.reset(inputs[0][:2].cpu().numpy(), inputs[0][2].item())
            loss = 0
            #breakpoint()
            for _ in range(SEQUENCE_LENGTH):
                if env.success:
                    break
                obs = torch.Tensor(env.get_vision()).to(self._device).reshape(1,1000)
                head_direction = one_hot(env.prev_move_global).to(self._device)

                pred = self._network([obs, head_direction])
                pred = pred["logits"]
                y = torch.tensor(env.find_optimal_move()).to(self._device).long()
                pred_step = torch.argmax(pred)
                #breakpoint()
                env.step(pred_step)
            correct += env.success
            grouped[targets.item()][1] += 1
            grouped[targets.item()][0] += env.success
            total += 1
        res = {}
<<<<<<< HEAD
        res["grouped"] = correct/total
=======
        _grouped = {str(k): v[0] / max(v[1], 1) for k, v in grouped.items()}
        _grouped.update({'num'+str(k): v[1] for k, v in grouped.items()})
        res["grouped"] = _grouped
>>>>>>> cdf5e5ef67844527a85a12010533bd9a9521fb1d
        res["top1"] = correct/total
        res["top10"] = correct
        res["top5"] = correct
        return res, res





    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0

            e_t = tqdm(enumerate(train_loader))
            for i, (_, inputs, targets) in e_t:

                # TODO Given the start and goal location, run exp.
                env = self.envs[str(targets.item())]

                env.reset(inputs[0][:2].cpu().numpy(), inputs[0][2].item())
                loss = 0
                #breakpoint()
                count = 0
                for _ in range(SEQUENCE_LENGTH):
                    if env.success:
                        break
                    obs = torch.Tensor(env.get_vision()).to(self._device).reshape(1,1000)
                    head_direction = one_hot(env.prev_move_global).to(self._device)

                    pred = self._network([obs, head_direction])
                    pred = pred["logits"]
                    y = torch.tensor(env.find_optimal_move()).to(self._device).long()
                    loss += F.cross_entropy(pred, y) #TODO add loss function
                    count += 1


                    pred_step = torch.argmax(pred)
                    #breakpoint()
                    env.step(pred_step)
                
                optimizer.zero_grad()
                loss_ewc = self.compute_ewc()
                loss = loss / count + lamda * loss_ewc
                loss.backward()
                optimizer.step()

                losses += loss.item()
                correct += int(env.success)
                e_t.set_postfix(correct = correct)
                total += 1
                #inputs, targets = inputs.to(self._device), targets.to(self._device)
                #logits = self._network(inputs)["logits"]
                #loss = F.cross_entropy(logits, targets)
                #optimizer.zero_grad()
                #loss.backward()
                #optimizer.step()
                #losses += loss.item()

                #_, preds = torch.max(logits, dim=1)
                #correct += preds.eq(targets.expand_as()).cpu().sum()
                #total += len(targets)

            scheduler.step()
            train_acc = np.around((correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)


    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            
            # TODO Given the start and goal location, run exp.
            env = self.envs[str(targets.item())]

            env.reset(inputs[0][:2].cpu().numpy(), inputs[0][2].item())
     
            loss = 0
            #breakpoint()
            for _ in range(SEQUENCE_LENGTH):

                if env.success:
                    break


                obs = torch.Tensor(env.get_vision()).to(self._device).reshape(1,1000)
                head_direction = one_hot(env.prev_move_global).to(self._device)

                pred = self._network([obs, head_direction])
                pred = pred["logits"]
                y = torch.tensor(env.find_optimal_move()).to(self._device).long()


                pred_step = torch.argmax(pred)
                #breakpoint()
                env.step(pred_step)
            
            correct += int(env.success)
            total += 1
        return np.around((correct) * 100 / total, decimals=2)



    def compute_ewc(self):
        loss = 0
        if len(self._multiple_gpus) > 1:
            for n, p in self._network.module.named_parameters():
                if n in self.fisher.keys():
                    loss += (
                        torch.sum(
                            (self.fisher[n])
                            * (p[: len(self.mean[n])] - self.mean[n]).pow(2)
                        )
                        / 2
                    )
        else:
            for n, p in self._network.named_parameters():
                if n in self.fisher.keys():
                    loss += (
                        torch.sum(
                            (self.fisher[n])
                            * (p[: len(self.mean[n])] - self.mean[n]).pow(2)
                        )
                        / 2
                    )
        return loss

    def getFisherDiagonal(self, train_loader):
        fisher = {
            n: torch.zeros(p.shape).to(self._device)
            for n, p in self._network.named_parameters()
            if p.requires_grad
        }
        self._network.train()
#        optimizer = optim.SGD(self._network.parameters(), lr=lrate)
        optimizer = optim.Adam(self._network.parameters(), lr=lrate)
        for i, (_, inputs, targets) in enumerate(train_loader):
            env = self.envs[str(targets.item())]
            #breakpoint()

            env.reset(inputs[0][:2].cpu().numpy(), inputs[0][2].item())
            loss = 0
            #breakpoint()
            for _ in range(SEQUENCE_LENGTH):
                if env.success:
                    break
                obs = torch.Tensor(env.get_vision()).to(self._device).reshape(1,1000)
                head_direction = one_hot(env.prev_move_global).to(self._device)

                pred = self._network([obs, head_direction])
                pred = pred["logits"]
                y = torch.tensor(env.find_optimal_move()).to(self._device).long()

                loss += F.cross_entropy(pred, y) #TODO add loss function

                pred_step = torch.argmax(pred)
                #breakpoint()
                env.step(pred_step)
            
            optimizer.zero_grad()
            loss.backward()



            #inputs, targets = inputs.to(self._device), targets.to(self._device)
            #logits = self._network(inputs)["logits"]
            #loss = torch.nn.functional.cross_entropy(logits, targets)
#            optimizer.zero_grad()
#            loss.backward()
            for n, p in self._network.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.pow(2).clone()
        for n, p in fisher.items():
            fisher[n] = p / len(train_loader)
            fisher[n] = torch.min(fisher[n], torch.tensor(fishermax))
        return fisher





if __name__ == "__main__":
    start = EWC()
    print(start.envs)

import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import DERNet, IncrementalNet
from utils.toolkit import count_parameters, target2onehot, tensor2numpy

##added
import sys
from watermaze.environment import *

def one_hot(num):
    """
    Given a number between 0-3, returns a one-hot tensor in PyTorch.
    """
    one_hot_vec = torch.zeros(4)
    one_hot_vec[num] = 1
    return one_hot_vec

##end added



EPSILON = 1e-8

init_epoch = 200
init_lr = 0.1
init_milestones = [60, 120, 170]
init_lr_decay = 0.1
init_weight_decay = 0.0005


epochs = 170
lrate = 0.1
milestones = [80, 120, 150]
lrate_decay = 0.1
batch_size = 128
weight_decay = 2e-4
num_workers = 8
T = 2

#added
SEQUENCE_LENGTH = 100

class DER(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._network = DERNet(args["convnet_type"], False)
        
        ##added
        NUM_OF_ENVS = 5

        self.envs = {}
        for i in np.arange(NUM_OF_ENVS):
            env_name = f"{i}" #to keep 1 index names
            self.envs[env_name] = SquareMaze(size = 15, observation_size = 1000, name = f"{i}")
        ##end added

    def after_task(self):
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = 4
        #self._known_classes + data_manager.get_task_size(
        #    self._cur_task
        #)
        self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        if self._cur_task > 0:
            for i in range(self._cur_task):
                for p in self._network.convnets[i].parameters():
                    p.requires_grad = False

        logging.info("All params: {}".format(count_parameters(self._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(self._network, True))
        )

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
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
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def train(self):
        self._network.train()
        if len(self._multiple_gpus) > 1 :
            self._network_module_ptr = self._network.module
        else:
            self._network_module_ptr = self._network
        self._network_module_ptr.convnets[-1].train()
        if self._cur_task >= 1:
            for i in range(self._cur_task):
                self._network_module_ptr.convnets[i].eval()

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._cur_task == 0: ##NOTE Possible ADAM
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                momentum=0.9,
                lr=init_lr,
                weight_decay=init_weight_decay,
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=init_milestones, gamma=init_lr_decay
            )
            if not self.args['skip']:
                self._init_train(train_loader, test_loader, optimizer, scheduler)
            else:
                test_acc = self._network.load_checkpoint(self.args)
                cur_test_acc = self._compute_accuracy(self._network, test_loader)
                logging.info(f"Loaded Test Acc:{test_acc} Cur_Test_Acc:{cur_test_acc}")
                
        else:
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=lrate,
                momentum=0.9,
                weight_decay=weight_decay,
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=milestones, gamma=lrate_decay
            )
            self._update_representation(train_loader, test_loader, optimizer, scheduler)
            if len(self._multiple_gpus) > 1:
                self._network.module.weight_align(
                    self._total_classes - self._known_classes
                )
            else:
                self._network.weight_align(self._total_classes - self._known_classes)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(init_epoch))
        for _, epoch in enumerate(prog_bar):
            self.train()
            losses = 0.0
            correct, total = 0, 0
            e_t = tqdm(enumerate(train_loader))
            for i, (_, inputs, targets) in e_t:
                # TODO Given the start and goal location, run exp.

                #breakpoint()
                env = self.envs[str(targets[0].item())]
                
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
                    y = y.view(1,)
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

            scheduler.step()
            train_acc = np.around(correct/total, decimals = 2)
#            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)

        logging.info(info)

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0,0
        
        e_t = tqdm(enumerate(loader))
        for i, (_, inputs, targets) in e_t:
            # TODO Given the start and goal location, run exp.

            #breakpoint()
            env = self.envs[str(targets[0].item())]
            
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
                count += 1

                pred_step = torch.argmax(pred)
                #breakpoint()
                env.step(pred_step)
            correct += int(env.success)
            total += 1
        return np.around(correct/total, decimals=2)
    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            self.train()
            losses = 0.0
            losses_clf = 0.0
            losses_aux = 0.0
            correct, total = 0, 0


            e_t = tqdm(enumerate(train_loader))
            for i, (_, inputs, targets) in e_t:

                # TODO Given the start and goal location, run exp.

                env = self.envs[str(targets[0].item())]
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
                    y = y.view(1,)
                    loss += F.cross_entropy(pred, y) #TODO add loss function
                    count += 1


                    pred_step = torch.argmax(pred)
                    #breakpoint()
                    env.step(pred_step)
                
                optimizer.zero_grad()
                loss = loss.item()
                losses_aux += loss.item() * 0
                losses_clf += loss.item()

                loss.backward()
                optimizer.step()

                losses += loss.item()
                correct += int(env.success)
                e_t.set_postfix(correct = correct)
                total += 1

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_aux {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    losses_aux / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_aux {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    losses_aux / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)

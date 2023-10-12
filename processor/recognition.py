#recognition_with_gradcam
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import argparse
import yaml
import numpy as np
import os
import seaborn as sns

# torch
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class
from draw_skeleton_fn import plot_rawdata, confusion_mat, draw_skel,draw_skel_true_pred,plot_heat_map
from .processor import Processor

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class REC_Processor(Processor):
    """
        Processor for Skeleton-based Action Recgnition
    """

    def load_model(self):
        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))
        self.model.apply(weights_init)
        self.loss = nn.CrossEntropyLoss()

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

    def adjust_lr(self):
        if self.arg.optimizer == 'SGD' and self.arg.step:
            lr = self.arg.base_lr * (
                0.1**np.sum(self.meta_info['epoch']>= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            self.lr = self.arg.base_lr

    def show_topk(self, k):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
        self.io.print_log('\tTop{}: {:.2f}%'.format(k, 100 * accuracy))

    def grad_cam_cal(self,activations,gradients):
        no_filter_lastlayer=gradients.size()[1]


        g_al=torch.mean(torch.mean(torch.mean(gradients,3),2),0)
        for i in range(no_filter_lastlayer):
            activations[:, i, :, :] *= g_al[i]


        cams = torch.mean(activations, 1).squeeze()
        cams=nn.functional.relu(cams)
        grad_cam_true=cams.data.cpu().numpy()
        return grad_cam_true



    def train(self):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []

        for data, label in loader:

            # get data
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)

            # forward
            output = self.model(data)
            loss = self.loss(output, label)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1

        self.epoch_info['mean_loss']= np.mean(loss_value)
        self.show_epoch_info()
        self.io.print_timer()

    def test(self, evaluation=True):

        self.model.eval()
        # print(self.model)
        loader = self.data_loader['test']
        loss_value = []
        result_frag = []
        label_frag = []
        tr_lab=[]
        pr_lab=[]

        iter=-1
        # action_name = np.load('action_name.npy')
        no_class = 60
        test_class_counts=np.zeros((60))

        for data_, label in loader:

            iter+=1
            sample_name=self.data_loader['test'].dataset.sample_name[iter].split('.')[0]
            # print(sample_name)
            # get data
            data_ = data_.float().to(self.dev)
            label = label.long().to(self.dev)
            true_label=label.data.cpu().numpy()
            test_class_counts[label-60]+=1  #average 275 datapoints


            data=data_
            output = self.model(data)

            result_frag.append(output.data.cpu().numpy())


            # get loss
            if evaluation:
                loss = self.loss(output, label)
                loss_value.append(loss.item())
                label_frag.append(label.data.cpu().numpy())

            #####STG-GradCAM
            values, indices = torch.max(output,1)
            true_label=label.data.cpu().numpy()
            pred_label=indices.cpu().numpy()


            tr_lab.append(label.data.cpu().numpy())
            pr_lab.append(indices.cpu().numpy())
            output[:,indices].backward()   #indices: pred label, label: true label

            gradients = self.model.get_activations_gradient()
            activations =  self.model.get_activations(self)
            grad_cam_true = grad_cam_cal(self,activations,gradients)

        self.result = np.concatenate(result_frag)

        if evaluation:
            self.label = np.concatenate(label_frag)
            self.epoch_info['mean_loss']= np.mean(loss_value)
            self.show_epoch_info()

            # show top-k accuracy


            for k in self.arg.show_topk:
                self.show_topk(k)

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        # region arguments yapf: disable
        # evaluation
        parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')
        # optim
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        # endregion yapf: enable

        return parser

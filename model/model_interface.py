# Copyright 2021 Zhongyang Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import torch
import importlib
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs

import pytorch_lightning as pl
from sklearn.metrics import f1_score, roc_auc_score

import clip


class MInterface(pl.LightningModule):
    def __init__(self, model_name, loss, lr, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()
        self.configure_loss()

    def forward(self, img):
        return self.model(img)

    def training_step(self, batch, batch_idx):
        image, text, label = batch["image"],batch["text"],batch["label"]
        if self.hparams.pretrain:
            #input = torch.cat((image,text),dim=-1)
            out = self(batch)
        else:
            out = self(batch)
            
        # one hot encoding

        label_one_hot = torch.zeros(label.shape[0], 6, device="cuda")
        label_one_hot.scatter_(1, label.long().unsqueeze(1), 1.0)
        label_binary = (label == 1).float().unsqueeze(1)
        positive = sum(label_binary[:,0])
        negative = len(label_binary[:,0]) - positive
        self.weight = negative/positive

        
        if self.hparams.binary:
            loss = self.loss_function(out, label_binary) 
            #auc = roc_auc_score(label_binary.detach().cpu().numpy(), out.detach().cpu().numpy())
            out = out>0
            correct_num = sum(label_binary[:,0] == out[:,0]).cpu().item()
            # calculate F1 and AUC
            f1 = f1_score(label_binary.detach().cpu().numpy(), out.detach().cpu().numpy())
            try:
                auc = roc_auc_score(label_binary.detach().cpu().numpy(), out.detach().cpu().numpy())
                self.log('train_auc', auc, on_step=True, on_epoch=True, prog_bar=False,logger=True)
            except:
                print('can\'t use auc')
            recall = f1_score(label_binary.detach().cpu().numpy(), out.detach().cpu().numpy(), average='macro')          
            #self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log('train_acc', correct_num/label.shape[0], on_step=True, on_epoch=True, prog_bar=False,logger=True)
            self.log('train_f1', f1, on_step=True, on_epoch=True, prog_bar=True,logger=True)
            self.log('train_recall', recall, on_step=True, on_epoch=True, prog_bar=False,logger=True)
            #self.log('positive', sum(out[:,0]), on_step=True, on_epoch=True, prog_bar=True)
        else:
            loss = self.loss_function(out, label_one_hot)
            out_digit = out.argmax(axis=1)
            correct_num = sum(label == out_digit).cpu().item()
            # 正例：类别非0
            positive_correct = ((out_digit != 0) & (label != 0)).sum().cpu().item()
            positive_total = (label != 0).sum().cpu().item()
            
            # 负例：类别为0
            negative_correct = ((out_digit == 0) & (label == 0)).sum().cpu().item()
            negative_total = (label == 0).sum().cpu().item()
            
            # 计算正负例的正确率
            positive_acc = positive_correct / positive_total if positive_total > 0 else 0
            negative_acc = negative_correct / negative_total if negative_total > 0 else 0
            self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log('train_acc', correct_num/label.shape[0], on_step=True, on_epoch=True, prog_bar=True)
            self.log('positive_acc', positive_acc, on_step=True, on_epoch=True, prog_bar=True)
            self.log('negative_acc', negative_acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        image, text, label = batch["image"],batch["text"],batch["label"]
        if self.hparams.pretrain:
            input = torch.cat((image,text),dim=-1)
            out = self(input)
        else:
            out = self(batch)
            

        label_one_hot = torch.zeros(label.shape[0], 6, device="cuda")
        label_one_hot.scatter_(1, label.long().unsqueeze(1), 1.0)
        label_binary = (label == 1).float().unsqueeze(1)
        positive = sum(label_binary[:,0])
        negative = len(label_binary[:,0]) - positive
        self.weight = negative/positive
        
        if self.hparams.binary:
            loss = self.loss_function(out, label_binary)
            out = out>0.5
            correct_num = sum(label_binary[:,0] == out[:,0]).cpu().item()
            f1 = f1_score(label_binary.detach().cpu().numpy(), out.detach().cpu().numpy())
            try:
                auc = roc_auc_score(label_binary.detach().cpu().numpy(), out.detach().cpu().numpy())
                self.log('test_auc', auc, on_step=True, on_epoch=True, prog_bar=False,logger=True)
            except:
                print('cnt\'t use auc')
            recall = f1_score(label_binary.detach().cpu().numpy(), out.detach().cpu().numpy(), average='macro')       
            #self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log('test_acc', correct_num/label.shape[0], on_step=True, on_epoch=True, prog_bar=False,logger=True)
            self.log('test_f1', f1, on_step=True, on_epoch=True, prog_bar=True,logger=True)
            self.log('test_recall', recall, on_step=True, on_epoch=True, prog_bar=False,logger=True)
        else:
            loss = self.loss_function(out, label_one_hot)
        
            out_digit = out.argmax(axis=1)

            correct_num = sum(label == out_digit).cpu().item()

            # 正例：类别非0
            positive_correct = ((out_digit != 0) & (label != 0)).sum().cpu().item()
            positive_total = (label != 0).sum().cpu().item()
            
            # 负例：类别为0
            negative_correct = ((out_digit == 0) & (label == 0)).sum().cpu().item()
            negative_total = (label == 0).sum().cpu().item()
            
            # 计算正负例的正确率
            positive_acc = positive_correct / positive_total if positive_total > 0 else 0
            negative_acc = negative_correct / negative_total if negative_total > 0 else 0
            
            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log('val_acc', correct_num/len(out_digit),
                    on_step=False, on_epoch=True, prog_bar=True)
            self.log('positive_acc', positive_acc, on_step=True, on_epoch=True, prog_bar=True)
            self.log('negative_acc', negative_acc, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def on_validation_epoch_end(self):
        # Make the Progress Bar leave there
        self.print('')

    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams.lr, weight_decay=weight_decay)

        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            if self.hparams.lr_scheduler == 'step':
                scheduler = lrs.StepLR(optimizer,
                                       step_size=self.hparams.lr_decay_steps,
                                       gamma=self.hparams.lr_decay_rate)
            elif self.hparams.lr_scheduler == 'cosine':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                  T_0=self.hparams.lr_decay_steps,
                                                  eta_min=self.hparams.lr_decay_min_lr)
            else:
                raise ValueError('Invalid lr_scheduler type!')
            return [optimizer], [scheduler]

    def configure_loss(self):
        loss = self.hparams.loss.lower()
        if self.hparams.binary:
            loss = 'bce'
        if loss == 'mse':
            self.loss_function = F.mse_loss
        elif loss == 'l1':
            self.loss_function = F.l1_loss
        elif loss == 'bce':
            pos_weight = torch.tensor([3],device="cuda")
            self.loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        elif loss == 'ce':
            weights = torch.tensor([0.15, 1.0, 1.0, 1.0, 1.0, 1.0], device="cuda")
            self.loss_function = torch.nn.CrossEntropyLoss(weight=weights)
        else:
            raise ValueError("Invalid Loss Type!")

    def convert_models_to_fp32(self,model): 
        for p in model.parameters(): 
            p.data = p.data.float() 
            p.grad.data = p.grad.data.float() 

    def load_model(self):
        name = self.hparams.model_name
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            Model = getattr(importlib.import_module(
                '.'+name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')
        self.model = self.instancialize(Model)

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)

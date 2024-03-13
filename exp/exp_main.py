import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO)

from exp.exp_basic import Exp_Basic
from models import EEGPred
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
from torch.utils.data import  DataLoader
from data.data_loader import Dataset_EEG_Data,Dataset_EEG_Data_Pred

import numpy as np
import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args,setting):
        super(Exp_Main, self).__init__(args,setting)

    def _build_model(self):
        model_dict = {
            'EEGPred' : EEGPred,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        ## multi_gpu
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag,scalers=None):
        args = self.args
        data_dict = {
            'EEG':Dataset_EEG_Data
        }
        Data = data_dict[self.args.data]
        if flag=='test':
            shuffle_flag = False; drop_last = False; batch_size = 1
        if flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1
            Data = Dataset_EEG_Data_Pred
        else:
            shuffle_flag = True; drop_last = False; batch_size = args.batch_size
   
        data_set = Data(
            path=os.path.join(args.root_path,
                            args.data_path
                            ),
            flag=flag,
            size=[args.seq_len, args.pred_len],
            target=args.target,
            scale=True,
            scalers=scalers,
            pred_true=args.pred_true
        )
        print(flag,len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _predict(self, batch_x, batch_y):
        def _run_model():
            outputs = self.model(batch_x)
            return outputs

        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = _run_model()
        else:
            outputs = _run_model()
        if self.flag=='pred':
            batch_y = None
        else:
            batch_y = batch_y[:,-self.args.pred_len:, :].to(self.device)
        return outputs, batch_y

    def vali(self, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                outputs, batch_y = self._predict(batch_x, batch_y)

                #for l in range(self.args.g_layers):
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                    #loss += loss

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        #self.model.train()
        return total_loss


    def train(self, setting):
        train_set, train_loader = self._get_data(flag='train')
        vali_set, vali_loader = self._get_data(flag='val', scalers=(train_set.scaler_x,train_set.scaler_y))
        test_set, test_loader = self._get_data(flag='test', scalers=(train_set.scaler_x,train_set.scaler_y))

        path = os.path.join( self.args.checkpoints, setting, 'Train')
        #path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                outputs, batch_y = self._predict(batch_x, batch_y)
                loss = criterion(outputs,batch_y)
                train_loss.append(loss.item())

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, criterion)
            test_loss = self.vali(test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)
           
        best_model_path = path + '/' + 'best_model.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        for name, param in self.model.named_parameters():
            print(f'Layer: {name}')
            print(param)
        return

    def test(self, setting, test=False):
        train_set, train_loader = self._get_data(flag='train')
        test_set, test_loader = self._get_data(flag='test', scalers=(train_set.scaler_x,train_set.scaler_y))

        path = os.path.join( self.args.checkpoints, setting, 'Train')
        if not os.path.exists(path):
            os.makedirs(path)

        if test:
            print('loading model')
            best_model_path = path + '/' + 'best_model.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                outputs, batch_y = self._predict(batch_x, batch_y)

                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()
                 #Plotting1
                if i % 5 ==0:
                    fig, ax = plt.subplots(3,1,figsize=[12, 8])
                    ax[0].plot(true[-1].reshape(-1, 1)[:6800])
                    ax[0].plot(pred[-1].reshape(-1, 1)[:6800])
                    ax[1].plot(true[-1].reshape(-1, 1)[6800*100:6800*101])
                    ax[1].plot(pred[-1].reshape(-1, 1)[6800*100:6800*101])
                    ax[2].plot(true[-1].reshape(-1, 1)[-6800:])
                    ax[2].plot(pred[-1].reshape(-1, 1)[-6800:])
                    # plt.legend(['Orig', 'Pred'])
                    # plt.title('Compare prediction')
                    plt.savefig(os.path.join(folder_path, f'compare_prediction_test_{i}.pdf'),bbox_inches='tight')
                preds.append(pred)
                trues.append(true)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        return


    def predict(self, setting, pred=False):
        train_set, train_loader = self._get_data(flag='train')
        pred_set,pred_loader = self._get_data(flag='pred', scalers=(train_set.scaler_x,train_set.scaler_y))

        path = os.path.join( self.args.checkpoints, setting, 'Train')
        if not os.path.exists(path):
            os.makedirs(path)

        if pred:
            print('loading model')
            best_model_path = path + '/' + 'best_model.pth'
            logging.info(best_model_path)
            self.model.load_state_dict(torch.load(best_model_path))
        
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y in pred_loader:
                batch_x = batch_x.float().to(self.device)
                
                batch_y =None
                outputs, batch_y = self._predict(batch_x, batch_y)

                
                pred = outputs.detach().cpu().numpy()
                
                return pred

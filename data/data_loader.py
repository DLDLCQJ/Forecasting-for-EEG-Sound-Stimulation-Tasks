import os
import scipy.io as sio
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import sklearn.preprocessing as prep

class Dataset_EEG_Data(Dataset):
    def __init__(self, path='/Users/simon/ky/EEG/Data/*.mat', flag='train', size=None, 
                 features='S', target='EpochData',scale=True, scalers=None,pred_true=False):
      
        self.seq_len = size[0]
        self.pred_len = size[1]
        self.flag = flag
        self.scale=scale
        self.scalers=scalers
        self.features = features
        self.target = target
        self.path = path
        self.pred_true = pred_true
        self.__read_data__()
        self.__split_data__()
        if flag == 'train' and not scalers:
            self.__fit_scalers__()
        elif scalers:
            self.scaler_x, self.scaler_y = scalers
            self.__fit_scalers__()
        else:
            raise ValueError("Scalers must be provided for non-training instance")

    def __read_data__(self):
  
        df_raw = []
        paths = glob(self.path)
        #path = paths[2:3]
        print(paths)
        for file in paths:
            mat_data = sio.loadmat(file)
            df_raw.append(mat_data[self.target][0][0][:564, :6800]) 
        df_raw = np.array(df_raw).reshape(-1, 1)
        self.df_raw = df_raw #if self.features == 'S' else df_raw[df_raw.columns[1:]]
        # ##plotting
        # folder_path = './plots/' 
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)
        # plt.figure(figsize = (12, 6))
        # plt.plot(self.df_raw, color = 'blue', label = 'EEG')
        # plt.title('All data')
        # plt.savefig(os.path.join(folder_path, 'eeg.pdf'),bbox_inches='tight')
        #
    def __split_data__(self):
        if self.pred_true:
            x, y = self.__slice_data__(self.df_raw[:-self.pred_len])
        else:
            x, y = self.__slice_data__(self.df_raw)
        ##
        total_len = int(len(x))
        tvt_end = total_len
        train_end = int(tvt_end * 0.5)
        if self.pred_true:
            self.train_x, self.train_y = x[:train_end], y[:train_end]
            self.val_x, self.val_y = x[train_end:], y[train_end:]
            train_inx = len(self.train_x)
            val_inx = train_inx + len(self.val_x)
        else:
            val_end = train_end + int(tvt_end * 0.25)
            test_end = val_end + int(tvt_end * 0.25)
            self.train_x, self.train_y = x[:train_end], y[:train_end]
            self.val_x, self.val_y = x[train_end:val_end], y[train_end:val_end]
            self.test_x, self.test_y = x[val_end:test_end], y[val_end:test_end]
            train_inx = len(self.train_x)
            val_inx = train_inx + len(self.val_x)
            test_inx = val_inx + len(self.test_x)
        # 
        if self.flag == 'train':
            self.inx = list(range(0, train_inx))
        elif self.flag == 'val':
            self.inx = list(range(train_inx,val_inx))
        elif self.flag == 'test':
            self.inx = list(range(val_inx, test_inx))
        else:
            raise ValueError("None pred")
    def __slice_data__(self,x):
        x_ ,y_= [],[]
        x = x.flatten() 
        L = len(x)
        for i in range(self.seq_len+self.pred_len, L, self.seq_len):
            tmp_x = x[i-self.pred_len-self.seq_len : i-self.pred_len]
            tmp_y = x[i-self.pred_len:i]
            x_.append(tmp_x)
            y_.append(tmp_y)
        return np.array(x_),np.array(y_)
        
    def __fit_scalers__(self):
        if self.flag == 'train' and not self.scalers:
            self.scaler_x = prep.MinMaxScaler(feature_range=(0, 1))
            self.scaler_y = prep.MinMaxScaler(feature_range=(0, 1))
            self.x_seqs = self.scaler_x.fit_transform(self.train_x.reshape(-1,self.seq_len))
            self.y_seqs = self.scaler_y.fit_transform(self.train_y.reshape(-1,self.pred_len))
        elif self.flag == 'val' and self.scalers:
            self.scaler_x, self.scaler_y = self.scalers
            self.x_seqs = self.scaler_x.transform(self.val_x.reshape(-1, self.seq_len))
            self.y_seqs = self.scaler_y.transform(self.val_y.reshape(-1, self.pred_len))
        elif self.flag == 'test' and self.scalers:
            self.scaler_x, self.scaler_y = self.scalers
            self.x_seqs = self.scaler_x.transform(self.test_x.reshape(-1, self.seq_len))
            self.y_seqs = self.scaler_y.transform(self.test_y.reshape(-1, self.pred_len))
        else:
            raise ValueError("Scalers must be provided for non-training instance")

    def __getitem__(self, index):
        seq_x = self.x_seqs[index].reshape(self.seq_len, -1)
        seq_y= self.y_seqs[index].reshape(self.pred_len,-1)
        return seq_x,seq_y
    
    def __len__(self):
        return len(self.inx)
        
    def inverse_transform(self, value):
        inv_value = self.scaler_y.inverse_transform(value)
        return inv_value
    


class Dataset_EEG_Data_Pred(Dataset):
    def __init__(self, path='/Users/simon/ky/EEG/Data/*.mat', flag='pred', size=None, 
                 features='S', target='EpochData',scale=True, scalers=None,pred_ture=True):
      
        self.seq_len = size[0]
        self.pred_len = size[1]
        self.flag = flag
        self.scale=scale
        self.scalers=scalers
        self.features = features
        self.target = target
        self.path = path
        self.pred_true = pred_ture
        self.scaler_x, self.scaler_y = scalers

        df_raw = []
        paths = glob(self.path)
        # path = paths[2:3]
        print(paths)
        for file in paths:
            mat_data = sio.loadmat(file)
            df_raw.append(mat_data[self.target][0][0][:564, :6800]) 
        df_raw = np.array(df_raw).reshape(-1, 1)
        #
        x_ = df_raw[-self.seq_len:]
        y_ = None
        pred_x = np.array(x_)
        self.x_seqs = self.scaler_x.transform(pred_x.reshape(-1, self.seq_len))
        self.y_seqs = None
    def __getitem__(self, index):
        seq_x = self.x_seqs[index].reshape(self.seq_len, -1)
        seq_y = None
        return seq_x,seq_y
    
    def __len__(self):
        return len(self.x_seqs)
        
    def inverse_transform(self, value):
        inv_value = self.scaler_y.inverse_transform(value)
        return inv_value

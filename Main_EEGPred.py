
import argparse
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

fix_seed = 2023
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EEGPred for Time Series Forecasting')
    # basic config
    parser.add_argument('--model_id', type=str,  default='EEGPred_2', help='model id') ##
    parser.add_argument('--model', type=str, default='EEGPred', help='model name')
    parser.add_argument('--data', type=str,  default='EEG', help='dataset type')
    parser.add_argument('--root_path', type=str, default='/lustre/project/eegprediction/Data', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='*.mat', help='data file')
    parser.add_argument('--target', type=str, default='EpochData', help='target feature in task')
    parser.add_argument('--checkpoints', type=str, default='/lustre/project/eegprediction/Results/checkpoints', help='location of model checkpoints')
    # forecasting task
    parser.add_argument('--seq_len', type=int, default=6800, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=6800*200, help='prediction sequence length')
    # model define  
    parser.add_argument('--u_layers', type=int, default=1, help='num of model layers')
    parser.add_argument('--c_out', type=int, default=1, help='input/output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=1, help='num of heads in attention mechanism')
    parser.add_argument('--num_layers', type=int, default=1, help='num of RNN layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--pred_true', action='store_true', help='inverse output data', default=False)
    # optimization
    parser.add_argument('--batch_size', type=int, default=6, help='batch size of train input data')
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='Exp', help='exp description') ##
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multile gpus')

    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    Exp = Exp_Main
   
    # setting record of experiments
    setting = '{}_{}_{}_{}_{}_{}'.format(
        args.model_id,
        args.model,
        args.data,
        args.seq_len,
        args.pred_len,
        args.d_model,
        )
    ##training
    exp = Exp(args,setting)  # set experiments
    # #
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)
    if args.pred_true:
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        pred = exp.predict(setting,True)
    else:
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting,True)

    torch.cuda.empty_cache()


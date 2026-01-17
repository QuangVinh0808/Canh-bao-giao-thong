import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from cfg import args
import numpy as np
from utils import metric

# quick local evaluator that does not depend on dataset.get_dataloader
args.seq_in_len = 1
args.seq_out_len = 1
args.train_ratio = 0.6
args.val_ratio = 0.2

DATA_PATH = os.path.join('data', args.dataset, f'{args.dataset}.npz')
print('Loading', DATA_PATH)
data = np.load(DATA_PATH)['data']  # shape (T, N, in_dim)
T, N, D = data.shape
seg_len = args.seq_in_len + args.seq_out_len
num_samples = T - seg_len + 1
if num_samples <= 0:
    raise RuntimeError('Not enough timesteps for seq lengths; reduce seq_in_len/seq_out_len')

# build samples
X, Y = [], []
for i in range(num_samples):
    x = data[i:i+args.seq_in_len]  # (seq_in_len, N, D)
    y = data[i+args.seq_in_len:i+seg_len, :, :args.out_dim]  # (seq_out_len, N, out_dim)
    X.append(x)
    Y.append(y)
X = np.stack(X, axis=0)  # (S, seq_in_len, N, D)
Y = np.stack(Y, axis=0)  # (S, seq_out_len, N, out_dim)

# split
train_end = int(args.train_ratio * X.shape[0])
val_end = int((args.train_ratio + args.val_ratio) * X.shape[0])
X_test = X[val_end:]
Y_test = Y[val_end:]

if X_test.shape[0] == 0:
    raise RuntimeError('No test samples after split — adjust train/val ratios')

all_mae, all_rmse, all_mape = [], [], []
for i in range(X_test.shape[0]):
    x = X_test[i]  # (seq_in_len, N, D)
    y = Y_test[i]  # (seq_out_len, N, out_dim)
    last = x[-1, :, 0]  # (N,) feature 0 is Flow_Volume
    pred = last.reshape(1, N, 1)  # (seq_out_len=1, N, out_dim=1)
    pred = np.expand_dims(pred, axis=0)  # (1,1,N,1)
    y = np.expand_dims(y, axis=0)  # (1,1,N,1)
    mae, rmse, mape = metric(pred, y)
    all_mae.append(mae)
    all_rmse.append(rmse)
    all_mape.append(mape)

print('Persistence baseline on test set:')
print('samples:', X_test.shape[0])
print('MAE =', float(np.mean(all_mae)))
print('RMSE =', float(np.mean(all_rmse)))
print('MAPE =', float(np.mean(all_mape)))

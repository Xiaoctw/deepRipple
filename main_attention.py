from model import *
from utils import *
import argparse
import warnings
import torch
from train import *

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()

parser.add_argument('--lr', type=float, default=3e-4,
                    help='The learning rate')
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--epochs', type=list, default=1000,
                    help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--dataset', type=str, default='facebook348')
parser.add_argument('--print_every', type=int, default=20)

args = parser.parse_args()
params = load_params(args.dataset)
adj_mat, ripple_mat = load_data(args.dataset)
N = adj_mat.shape[0]
pco_mat = random_surfing(adj_mat, epochs=params['surfing_epoch'], alpha=params['alpha'])
ppmi_mat = PPMI_matrix(pco_mat)
ppmi_mat = torch.from_numpy(ppmi_mat).float()
ripple_mat = torch.from_numpy(ripple_mat).float()
GPU = args.cuda and torch.cuda.is_available()
lr = params["lr"]
epochs = params['epochs']
model = DeepRipple(input_dim=N, hidden_dims=params['hidden_dims'], output_dim=params['output_dim'],)


# 训练得到模型，该模型为采用了attention的自编码器模型
if __name__ == '__main__':
    if GPU:
        model = model.cuda()
    print('节点数量:{}'.format(N))
    print(ppmi_mat.shape)
    # print(ripple_mat.shape)
    train(model, ppmi_mat, ripple_mat,b=10, epochs=epochs, lr=lr, batch_size=args.batch_size, print_every=args.print_every,
          GPU=GPU)
    model = model.cpu()
    out1, out2 = model(ppmi_mat[:20], ripple_mat[:20])
    print('ppmi:')
    print(out1[0][:15])
    print(ppmi_mat[0][:15])
    # print(out2)
    print('---------------')
    print('ripple_mat:')
    print(out2[0][:15])
    print(ripple_mat[0][:15])

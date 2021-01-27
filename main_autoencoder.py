from model import *
from utils import *
import argparse
import warnings
import torch
from train import *

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()

parser.add_argument('--lr', type=float, default=6e-4,
                    help='The learning rate')
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--epochs2', type=list, default=1000,
                    help='Number of epochs to train.')
parser.add_argument('--epochs1', type=list, default=600,
                    help='Number of epochs to train.')
parser.add_argument('--zero_ratio', type=float, default=0.4, help='The probability of random 0.')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--dataset', type=str, default='usa-airports')
parser.add_argument('--print_every', type=int, default=40)
parser.add_argument('--w', type=float, default=0.5)
parser.add_argument('--ratio', type=int, default=5)

args = parser.parse_args()
params = load_params(args.dataset)
adj_mat, ripple_mat = load_data(args.dataset)
N = adj_mat.shape[0]
pco_mat = random_surfing(adj_mat, epochs=params['surfing_epoch'], alpha=params['alpha'])
ppmi_mat = PPMI_matrix(pco_mat)
ppmi_mat=helper(ppmi_mat)
M = args.w * ppmi_mat + (1 - args.w) * ripple_mat
# ppmi_mat = torch.from_numpy(ppmi_mat).float()
# ripple_mat = torch.from_numpy(ripple_mat).float()
M = torch.from_numpy(M).float() * args.ratio
GPU = args.cuda and torch.cuda.is_available()
lr = params["lr"]
epochs = params['epochs']
model = StackAutoEncoder(input_dim=N, hidden_dims=params['hidden_dims'], output_dim=params['output_dim'],
                         zero_ratio=args.zero_ratio)
model_path = Path(__file__).parent / 'models' / (
            'AutoEncoder_' + "w:{}_".format(args.w) + "ratio:{}_".format(args.ratio) + args.dataset + '.pkl')

# 训练得到模型，该模型为堆叠的自编码器模型，自编码器的输入为
if __name__ == '__main__':
    if GPU:
        model = model.cuda()
    print('节点数量:{}'.format(N))
    print(ppmi_mat.shape)
    # print(ripple_mat.shape)
    train_autoEncoder(model, M, b=3,epochs1=args.epochs1,
                      epochs2=epochs, lr=lr, batch_size=args.batch_size, print_every=args.print_every,
                      GPU=GPU)
    model = model.cpu()
    model.eval()
    torch.save(model, model_path)
    out = model(M[:20])
    print('M:')
    print(M[0][:15])
    print('out:')
    print(out[0][:15])

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
parser.add_argument('--dataset', type=str, default='europe-airports')
parser.add_argument('--w', type=float, default=0.5)
parser.add_argument('--ratio', type=int, default=5)


args = parser.parse_args()
params = load_params(args.dataset)
adj_mat, ripple_mat = load_data(args.dataset)
N = adj_mat.shape[0]
surfing_mat = random_surfing(adj_mat, epochs=params['surfing_epoch'], alpha=params['alpha'])
# ppmi_mat = PPMI_matrix(pco_mat)
# ppmi_mat=helper(ppmi_mat)
M = args.w * surfing_mat + (1 - args.w) * ripple_mat
# ppmi_mat = torch.from_numpy(ppmi_mat).float()
# ripple_mat = torch.from_numpy(ripple_mat).float()
M = torch.from_numpy(M).float() * args.ratio
model_path = Path(__file__).parent / 'models' / (
        'AutoEncoder_' + "w:{}_".format(args.w) + "ratio:{}_".format(args.ratio) + args.dataset + '.pkl')
save_path = Path(__file__).parent / 'embedding_results' / ('{}_outVec.txt'.format(args.dataset))

# 导入训练好的模型，对图进行嵌入
if __name__ == '__main__':
    model = torch.load(model_path)
    model.eval()
    embs = model.emb(M)
    embs = embs.cpu().detach().numpy()
    out = model(M[:20])
    print('M:')
    print(M[0][:15])
    print('out:')
    print(out[0][:15])
    np.savetxt(save_path, embs)

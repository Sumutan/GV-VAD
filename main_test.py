import time

from torch.utils.data import DataLoader
import torch.optim as optim
from model_tcc_triplet_v2_oral import Model
from dataset import Dataset
from test_10crop_tcc import test
import option
from utils import *
from config import *
import torch, sys
from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score, confusion_matrix
import numpy as np
from utils import get_gt, anomap, compute_auc, compute_far
import pickle

def pad_array(arr, length):  # padding 1-D ndarray by last element
    last_element = arr[-1]
    padding_length = length - len(arr)
    padding = [last_element] * padding_length
    return np.pad(arr, (0, padding_length), 'constant', constant_values=(0, last_element))


def get_gt_dic(picklePath):
    with open(picklePath, 'rb') as f:
        frame_label = pickle.load(f)
    return frame_label


if __name__ == '__main__':
    args = option.parser.parse_args()
    config = Config(args)
    seed_everything(args.seed)
    test_loader = DataLoader(Dataset(args, test_mode=True),
                             batch_size=1, shuffle=False,
                             num_workers=0, pin_memory=False)

    model = Model(args)
    model.load_state_dict(
        torch.load('/media/cw/584485FC4485DD5E/csh/LAP/ckpt/save/ucfg2-LAP-seed0.pkl'))
    # model.load_state_dict(
    #     torch.load('/home/tcc/下载/test/ckpt/ucfcrime-both-text_agg-concat-0.0001-extra_loss-65-228-uca-89_87.pkl'))
    # model.load_state_dict(
    #     torch.load('/home/tcc/下载/test/ckpt/ucf_gamma0_1_90_45_large.pkl'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if not os.path.exists('./ckpt'):
        os.makedirs('./ckpt')

    optimizer = optim.Adam(model.parameters(),
                           lr=config.lr[0], weight_decay=0.005)

    best_AUC = -1
    output_path = 'output/test'  # put your own path here
    if args.use_dic_gt:
        if 'shanghai' in args.test_rgb_list:
            gt_dic = get_gt_dic('./list/gt-sh2-dic.pickle')
        elif 'ucf' in args.test_rgb_list:
            gt_dic = get_gt_dic('./list/gt-ucf-dic.pickle')
        elif 'violence' in args.test_rgb_list:
            gt_dic = get_gt_dic('./list/gt-violence-dic.pickle')
        elif 'tad' in args.test_rgb_list:
            gt_dic = get_gt_dic('./list/gt-tad-dic.pickle')
        else:
            raise ValueError('Dataset not supported')
    with torch.no_grad():
        model.eval()
        pred, gt = torch.zeros(0), torch.zeros(0)
        pred_abn, gt_abn = torch.zeros(0), torch.zeros(0)  # 仅仅记录异常视频的预测结果与GT

        all_count, abn_count = 0, 0  # 正常/异常视频计数器
        predict_dict, gt_dict = {}, {}
        clip_features = {}
        lat_features = {}
        skip = 0

        duras = 0

        for i, (input, text, fname) in enumerate(test_loader):  # test set has 199 videos
            input = input.to(device)
            input = input.permute(0, 2, 1, 3)
            text = text.to(device)
            text = text.permute(0, 2, 1, 3)
            # input.shape = (1,10,T,2048); T clips, each clip has 16frames, each frame has 10 crops
            # https://github.com/tianyu0207/RTFM/issues/51
            # 使用时可以把10那一维拉平变成(1, 10*T, 2048), 中间那一维就是visual features再和caption进行concat操作
            start = time.time()
            clip_feature, lat_feature, score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, \
            feat_select_normal_bottom, logits, scores_nor_bottom, scores_nor_abn_bag, feat_magnitudes \
                = model(input, text, fname[0])
            dura = time.time() - start
            duras += dura
            print(f"time for {input.shape[2]}  -> {dura}")
            # 注意这里的score_abnormal和score_normal是一维的，是每一个video的一个分数，而logits则是一个T维的vector给每一个snippet都打了分

            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)
            sig = logits
            fname = fname[0]
            if fname[-4:] == 'clip':
                fname = fname[:-5]
            if fname[-8:] == 'videomae':
                fname = fname[:-9]
            if args.dataset == "shanghai" or args.dataset == "ucfcrime" or args.dataset == "violence":
                _gt_raw = gt_dic[fname]
            elif args.dataset == "tad":
                _gt_raw = gt_dic[fname + '.mp4']
            if sig.shape[0] * 16 < len(_gt_raw):
                _gt = torch.tensor(_gt_raw[:len(sig) * 16])
            else:
                _gt = torch.tensor(pad_array(_gt_raw, len(sig) * 16))
            # if args.alignment_method == 'add':  # TEVAD default, len(pred*16)>=len(gt)
            #     # 由于pred的策略是对最后不足16frame的视频以最后一个frame进行补齐，所以此处也对应的对gt进行补齐
            #     _gt = torch.tensor(pad_array(_gt_raw, len(sig) * 16))
            # elif args.alignment_method == 'cut':  # 对于最后不足16的frame直接丢弃不做推理，len(pred*16)<=len(gt)
            #     _gt = torch.tensor(_gt_raw[:len(sig) * 16])
            # else:
            #     raise ValueError('Error: alignment method not specified')

            pred = torch.cat((pred, sig))  # pred means pread_all_preds
            gt = torch.cat((gt, _gt))
            all_count += 1
            if np.max(_gt_raw):  # 如果是异常视频（视频中有1p为1），拼接到异常列表中用于AUC_abn/far_abn计算
                abn_count += 1
                skip += 1
                pred_abn = torch.cat((pred_abn, sig))
                gt_abn = torch.cat((gt_abn, _gt))
                clip_features[fname] = clip_feature.cpu().numpy()
                lat_features[fname] = lat_feature.cpu().numpy()

            predict_dict[fname] = np.repeat(np.squeeze(sig.cpu().numpy()), 16)
            gt_dict[fname] = _gt.cpu().numpy()

        gt = gt.cpu().numpy()
        gt_abn = gt_abn.cpu().numpy()
        pred = list(pred.cpu().detach().numpy())
        print(duras / len(pred))
        pred = np.repeat(np.array(pred), 16)  # 数组中的每个元素重复16遍，即同一个clip中的16帧共享相同的预测结果
        pred_abn = list(pred_abn.cpu().detach().numpy())
        pred_abn = np.repeat(np.array(pred_abn), 16)  # 数组中的每个元素重复16遍，即同一个clip中的16帧共享相同的预测结果

        ap = average_precision_score(list(gt), pred)
        print('ap : ' + str(ap))
        rec_auc_all, pr_auc_all, fpr_all, tpr_all = compute_auc(gt, pred, 'all')
        far_all = compute_far(gt, pred, 'all')
        with open('result.pickle', 'wb') as file:
            pickle.dump(predict_dict, file)

        if args.use_dic_gt:
            rec_auc_abn, pr_auc_abn, fpr_all, tpr_all = compute_auc(gt_abn, pred_abn, 'abnormal')
            far_abn = compute_far(gt_abn, pred_abn, 'abnormal')

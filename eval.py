import pickle
import os
import numpy as np
import sklearn.metrics
from sklearn.metrics import roc_auc_score, confusion_matrix,roc_curve
import sys
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from utils import compute_auc

def anomap(predict_dict, label_dict, save_path, itr, save_root, zip=False):
    """

    :param predict_dict:
    :param label_dict:
    :param save_path:
    :param itr:
    :param zip: boolen, whether save plots to a zip
    :return:
    """
    if os.path.exists(os.path.join(save_root, save_path, 'plot')) == 0:
        os.makedirs(os.path.join(save_root, save_path, 'plot'))
    for k, v in predict_dict.items():
        # if k == '01_028' or k == '01_047' or k == '01_074' or k == '03_002' or k == '04_002' or k == '05_043' or k == '05_044' or k == '05_045' \
        #         or k == '05_043' or k == '08_001' or k == '08_006' or k == '08_007' or k == '08_010' or k == '08_032':
        #     v = v[:-1]
        predict_np = v.repeat(16)
        label_np = label_dict[k][:len(v.repeat(16))]
        x = np.arange(len(predict_np))
        plt.figure(figsize=(12.06, 3.4))
        plt.plot(x, predict_np, color='royalblue', label='Ours', linewidth=2)
        #plt.plot(x, predict_memae, color='green', label='MemAE',linestyle='--', linewidth=2)np.save('images/05_26/' + k + '_1', predict_np)
        #np.save('images/05_26/' + k, label_np)
        #plt.plot(x, predict_np, color='b', linewidth=1)
        #print(type(label_np))
        if type(label_np) is list:
            label_np = np.array(label_np)
        plt.fill_between(x, label_np, where=label_np > 0, facecolor="pink",alpha=0.8)
        plt.yticks(np.arange(0.1, 1.1, step=0.1),weight='bold',size=15)
        plt.xticks(weight='bold',size=15)
        plt.xlabel('Frames',fontsize=15,fontweight='bold')
        plt.ylabel('Anomaly scores',fontsize=15,fontweight='bold')
        #plt.grid(True, linestyle='-.')
        plt.legend()
        # plt.show()
        if os.path.exists(os.path.join(save_root, save_path, 'plot', 'itr_{}'.format(itr))) == 0:
            os.makedirs(os.path.join(save_root, save_path, 'plot', 'itr_{}'.format(itr)))
            plt.savefig(os.path.join(save_root, save_path, 'plot', 'itr_{}'.format(itr), k))
        else:
            plt.savefig(os.path.join(save_root, save_path, 'plot', 'itr_{}'.format(itr), k))
        plt.close()


def scorebinary(scores=None, threshold=0.5):
    scores_threshold = scores.copy()
    scores_threshold[scores_threshold < threshold] = 0
    scores_threshold[scores_threshold >= threshold] = 1
    return scores_threshold


def eval_p(predict_dict,plot=False):
    global label_dict_path
    with open(file="list/gt-ucf-dic.pickle", mode='rb') as f:
        frame_label_dict = pickle.load(f)
        if type(frame_label_dict) is list:
            frame_label_dict = np.array(frame_label_dict)
    gt,pred = np.zeros(0),np.zeros(0)
    for k in predict_dict:
        gt = np.concatenate((gt,frame_label_dict[k]),axis=0)
        pred = np.concatenate((pred,predict_dict[k]),axis=0)
        gt = gt[:len(pred)]
        pred = pred[:len(gt)]
    rec_auc_all, pr_auc_all, fpr_all, tpr_all = compute_auc(gt, pred, 'all')
    #print("AUC:",rec_auc_all)

def calculate_eer(y,y_score):
    fpr,tpr,thresholds = roc_curve(y,y_score,pos_label=1)
    eer = brentq(lambda x:1. - x - interp1d(fpr,tpr)(x), 0., 1.)
    auc = sklearn.metrics.auc(fpr,tpr)
    thresh = interp1d(fpr,thresholds)(eer)
    return eer, auc, fpr,tpr

def draw_ROC(auc, fpr, tpr, save_path, itr):
    plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.2f})'.format(auc), lw=2)
    plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    # save roc curve
    np.savez(r'{}/plot/ROC.npz'.format(save_path), auc, fpr, tpr)
    plt.savefig(os.path.join(save_path, 'plot', 'itr_{}'.format(itr)))
    plt.close()

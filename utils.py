import pickle

import visdom
import numpy as np
import torch
import random, os


class Visualizer(object):
    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        self.index = {}

    def plot_lines(self, name, y, **kwargs):
        '''
        self.plot('loss', 1.00)
        '''
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=str(name),
                      opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1

    def disp_image(self, name, img):
        self.vis.image(img=img, win=name, opts=dict(title=name))

    def lines(self, name, line, X=None):
        if X is None:
            self.vis.line(Y=line, win=name)
        else:
            self.vis.line(X=X, Y=line, win=name)

    def scatter(self, name, data):
        self.vis.scatter(X=data, win=name)


def process_feat(feat, length):
    new_feat = np.zeros((length, feat.shape[1])).astype(np.float32)

    r = np.linspace(0, len(feat), length + 1, dtype=np.int)  # len=33,存入要取的frame index
    for i in range(length):
        if r[i] != r[i + 1]:
            new_feat[i, :] = np.mean(feat[r[i]:r[i + 1], :], 0)  # r[i]:r[i+1]这些feat求平均
        else:
            new_feat[i, :] = feat[r[i], :]  # 不足32帧补全
    return new_feat


def minmax_norm(act_map, min_val=None, max_val=None):
    if min_val is None or max_val is None:
        relu = torch.nn.ReLU()
        max_val = relu(torch.max(act_map, dim=0)[0])
        min_val = relu(torch.min(act_map, dim=0)[0])

    delta = max_val - min_val
    delta[delta <= 0] = 1
    ret = (act_map - min_val) / delta

    ret[ret > 1] = 1
    ret[ret < 0] = 0

    return ret


def modelsize(model, input, type_size=4):
    # check GPU utilisation
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))

    input_ = input.clone()
    input_.requires_grad_(requires_grad=False)

    mods = list(model.modules())
    out_sizes = []

    for i in range(1, len(mods)):
        m = mods[i]
        if isinstance(m, nn.ReLU):
            if m.inplace:
                continue
        out = m(input_)
        out_sizes.append(np.array(out.size()))
        input_ = out

    total_nums = 0
    for i in range(len(out_sizes)):
        s = out_sizes[i]
        nums = np.prod(np.array(s))
        total_nums += nums

    print('Model {} : intermedite variables: {:3f} M (without backward)'
          .format(model._get_name(), total_nums * type_size / 1000 / 1000))
    print('Model {} : intermedite variables: {:3f} M (with backward)'
          .format(model._get_name(), total_nums * type_size * 2 / 1000 / 1000))


def save_best_record(test_info, file_path, metrics):
    fo = open(file_path, "w")
    fo.write("epoch: {}\n".format(test_info["epoch"][-1]))
    fo.write(metrics + ": " + str(test_info[metrics][-1]) + '\n')
    fo.write('test_AUC_all' + ": " + str(test_info["test_AUC"][-1]) + '\n')
    fo.write('test_AUC_abn' + ": " + str(test_info["test_AUC_abn"][-1]) + '\n')
    fo.write('far_all' + ": " + str(test_info["test_far_all"][-1])+'\n')
    fo.write("far_abnormal" + ": " + str(test_info["test_far_abn"][-1])+'\n')
    fo.write("AP" + ": " + str(test_info["test_AP"][-1]) + '\n')
    fo.close()


def vid_name_to_path(vid_name, mode):  # TODO: change absolute paths! (only used by visual codes)
    root_dir = '/home/acsguser/Codes/SwinBERT/datasets/Crime/data/'
    types = ["Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting", "RoadAccidents", "Robbery",
             "Shooting", "Shoplifting", "Stealing", "Vandalism"]
    for t in types:
        if vid_name.startswith(t):
            path = root_dir + t + '/' + vid_name
            return path
    if vid_name.startswith('Normal'):
        if mode == 'train':
            path = root_dir + 'Training_Normal_Videos_Anomaly/' + vid_name
        else:
            path = root_dir + 'Testing_Normal_Videos_Anomaly/' + vid_name
        return path
    raise Exception("Unknown video type!!!")


def seed_everything(seed=4869):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # torch.backends.cudnn.deterministic = True：设置 cuDNN 库的随机数生成器行为为确定性模式。在这种模式下，cuDNN
    # 库的算法将使用确定性的算法，以便于在不同的运行环境下获得相同的结果。这通常用于保证实验的可重复性。需要注意的是，启用确定性模式可能会降低性能。
    # torch.backends.cudnn.benchmark = False：禁用cuDNN 库的自动调整算法。在默认情况下，cuDNN库会根据输入数据的大小和其他参数自动选择最
    # 佳的卷积算法，以便获得最佳的性能。但是，这种自动调整算法可能会导致不同的运行环境下获得不同的结果，因此如果要保证实验的可重复性，通常需要禁用该算法。
    # 因此，这两行代码的作用是为了保证PyTorch 在使用 cuDNN 库时的可重复性，以便于在不同的运行环境下获得相同的结果。但需要注意，启用确定性模式可
    # 能会降低性能，而禁用自动调整算法可能会导致性能下降。因此，在实际使用中需要根据具体情况进行权衡。


def get_rgb_list_file(ds, is_test, feat_extractor='i3d'):
    if feat_extractor not in ['i3d', 'videoMAE', 'clip']:
        raise ValueError("feat_extractor should be i3d,videoMAE or clip")

    if "ucf" in ds:
        ds_name = "Crime"
        if feat_extractor == 'i3d':
            if is_test:
                rgb_list_file = 'list/ucf-i3d-test.list'
            else:
                rgb_list_file = 'list/ucf-i3d.list'
        elif feat_extractor == 'videoMAE':
            if is_test:
                rgb_list_file = 'list/ucf-videoMAE-test.list'
            else:
                rgb_list_file = 'list/ucf-videoMAE.list'
        elif feat_extractor == 'clip':
            raise NotImplementedError
    elif "shanghai" in ds:
        ds_name = "Shanghai"
        if feat_extractor == 'i3d':
            if is_test:
                rgb_list_file = 'list/shanghai-i3d-test-10crop.list'
            else:
                rgb_list_file = 'list/shanghai-i3d-train-10crop.list'
        elif feat_extractor == 'clip':
            if is_test:
                rgb_list_file = 'list/shanghai-clip-test-10crop.list'
            else:
                rgb_list_file = 'list/shanghai-clip-train-10crop.list'
        else:
            raise NotImplementedError
    elif "violence" in ds:
        ds_name = "Violence"
        if is_test:
            rgb_list_file = 'list/violence-i3d-test.list'
        else:
            rgb_list_file = 'list/violence-i3d.list'
    elif "ped2" in ds:
        ds_name = "UCSDped2"
        if is_test:
            rgb_list_file = 'list/ped2-i3d-test.list'
        else:
            rgb_list_file = 'list/ped2-i3d.list'
    elif "TE2" in ds:
        ds_name = "TE2"
        if is_test:
            rgb_list_file = 'list/te2-i3d-test.list'
        else:
            rgb_list_file = 'list/te2-i3d.list'
    else:
        raise ValueError("dataset should be either ucf, shanghai, or violence")
    return ds_name, rgb_list_file


def get_gt(ds, gt_file):
    if gt_file is not None:
        gt = np.load(gt_file)
    else:
        if 'shanghai' in ds:
            gt = np.load('list/gt-sh2.npy')
        elif 'ucf' in ds:
            gt = np.load('list/gt-ucf.npy')
        elif 'violence' in ds:
            gt = np.load('list/gt-violence.npy')
        elif 'ped2' in ds:
            gt = np.load('list/gt-ped2.npy')
        elif 'TE2' in ds:
            gt = np.load('list/gt-te2.npy')
        else:
            raise Exception("Dataset undefined!!!")
    return gt


from sklearn.metrics import auc, roc_curve, precision_recall_curve, confusion_matrix


def compute_auc(gt, pred, printname=None):  # 计算 ROC 曲线和 AUC
    fpr, tpr, threshold = roc_curve(list(gt), pred)  # 计算 ROC 曲线
    rec_auc = auc(fpr, tpr)  # 计算 ROC 曲线下的面积（AUC）
    precision, recall, th = precision_recall_curve(list(gt), pred)  # 计算 PR 曲线
    pr_auc = auc(recall, precision)  # 计算 PR 曲线下的面积（PR AUC）

    if printname:
        print(f'rec_auc_{printname} : ' + str(rec_auc))
        #print(f'pr_{printname} : ' + str(pr_auc))

    return rec_auc, pr_auc, fpr, tpr


def compute_far(gt, pred, printname=None):
    preTrue = [1 if x > 0.5 else 0 for x in pred]  # 将预测标签转换为二分类的 0 或 1
    tn, fp, fn, tp = confusion_matrix(gt, preTrue).ravel()  # 计算混淆矩阵中的 TP，TN，FP，FN 值
    far = fp / (fp + tn)  # 计算 FAR
    if printname:
        print(f'far_{printname} : ' + str(far))
    return far


import matplotlib.pyplot as plt


def anomap(predict_dict, label_dict, save_path: str, itr, save_root: str, zip=False):
    # 绘制异常检测模型的预测结果
    # predict_dict: 预测结果字典，格式为{k: v}
    # label_dict: 标签字典，格式为{k: v}
    # save_path: 结果保存路径
    # itr: 迭代次数
    # save_root: 结果保存根目录
    # zip: 是否将结果打包成zip文件保存，默认为False
    os.mkdir(save_root)
    with open(os.path.join(save_root,'result_itr{}.pickle'.format(itr)),'wb') as file:
        pickle.dump(predict_dict,file)
    os.makedirs(os.path.join(save_root, 'plot'), exist_ok=True)  # 如果保存结果的目录不存在，则创建目录
    # 如果zip为True，则将结果打包成zip文件保存
    if zip:
        raise NotImplementedError
        # 设置zip文件名
        zip_file_name = os.path.join(save_root, save_path, 'plot', 'itr_{}.zip'.format(itr))
        with zipfile.ZipFile(zip_file_name, mode="w") as zf:
            # 对于每个预测结果，绘制曲线并保存到zip文件中
            for k, v in predict_dict.items():
                img_name = k + '.jpg'
                predict_np = v.repeat(16)
                label_np = label_dict[k][:len(v.repeat(16))]
                x = np.arange(len(predict_np))
                plt.plot(x, predict_np, label='Anomaly scores', color='darkgreen', linewidth=1)
                plt.fill_between(x, label_np, where=label_np > 0, facecolor="red", alpha=0.3)
                plt.yticks(np.arange(0, 1.1, step=0.1))
                plt.xlabel('Frames')
                plt.grid(True, linestyle='-.')
                plt.legend()
                # 将绘制的结果保存到缓存中
                buf = io.BytesIO()
                plt.savefig(buf)
                plt.close()
                # 将缓存中的结果保存到zip文件中
                zf.writestr(img_name, buf.getvalue())

    else:
        # 对于每个预测结果，绘制曲线并保存到指定的路径中
        for k, v in predict_dict.items():
            # predict_np = v.repeat(16)
            predict_np = v
            # label_np = label_dict[k][:len(v.repeat(16))]
            label_np = label_dict[k]
            x = np.arange(len(predict_np))
            #plt.plot(x, predict_np, color='blue', label='predicted scores', linewidth=2)
            plt.plot(x, predict_np, color='blue', label='predicted scores', linewidth=2)
            plt.fill_between(x, label_np, where=label_np > 0, facecolor="red", alpha=0.3)
            plt.yticks(np.arange(0, 1.1, step=0.1))
            plt.xlabel('Frames')
            plt.ylabel('Anomaly scores')
            #plt.grid(True, linestyle='-.')
            #plt.legend()
            # plt.show()

            os.makedirs(os.path.join(save_root, save_path, 'plot', 'itr_{}'.format(itr)), exist_ok=True)
            plt.savefig(os.path.join(save_root, save_path, 'plot', 'itr_{}'.format(itr), k+'.png'))
            plt.close()

        #print("curve save to {}".format(os.path.join(save_root, save_path, 'plot', 'itr_{}'.format(itr))))

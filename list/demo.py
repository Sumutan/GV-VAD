import pickle
def get_gt_dic(picklePath):
    with open(picklePath, 'rb') as f:
        frame_label = pickle.load(f)
    return frame_label
gt_dic = get_gt_dic('gt-tad-dic.pickle')
pass
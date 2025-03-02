from eval import eval_p
from torch.utils.data import Dataset, DataLoader
import time
import torch
import os
import argparse
import numpy as np
import pickle
import json

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = 'plot'
    with open('result_itr260.pickle','rb') as file:
        result = pickle.load(file)
    #eval_p(predict_dict=result, plot=False)
    test_list = ['Abuse','Arrest','Arson','Assault','Burglary','Explosion','Fighting','RoadAccidents','Robbery','Shooting','Shoplifting','Stealing','Vandalism']
        #draw plots
    for j in range(len(test_list)):
        name = test_list[j]

        result_new = {}
        for key in result.keys():
            if name in key:
                result_new[key] = result[key]
        print('AUC of ',name,':')
        eval_p(predict_dict=result_new, plot=False)



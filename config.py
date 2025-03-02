import numpy as np
import os


class Config(object):  # 用于存储模型的超参数和其他配置信息,以及训练中的全局动态参数
    def __init__(self, args):
        self.lr = eval(args.lr)  # 用于将字符串作为 Python 表达式进行求值，并返回求值结果
        self.lr_str = args.lr
        self.training_step=0

    def __str__(self):  # 用于将配置信息格式化为字符串
        attrs = vars(self)  # vars(self) 函数用于获取对象的属性字典，返回一个字典对象，其中包含对象的所有属性名称和属性值。
        attr_lst = sorted(attrs.keys())
        return '\n'.join("- %s: %s" % (item, attrs[item]) for item in attr_lst if item != 'lr')

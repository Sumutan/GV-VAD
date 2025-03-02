import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle
import random
def draw_abn(abn_index,gt,name):
    # 创建一个示例的标签数组，0表示天蓝色，1表示粉色
    labels = gt

    # 创建一个随机的0和1组成的数组
    data = abn_index

    # 创建决策边界
    fig, ax = plt.subplots()
    diagonal_x = [0,10]
    diagonal_y = [10, 0]
    ax.plot(diagonal_x,diagonal_y,color='black',linewidth=3)
    blue_rectan = patches.Polygon([[0,0],[10,0],[0,10]],closed=True,facecolor='lightskyblue')
    pink_rectan = patches.Polygon([[10,0], [10, 10], [0, 10]], closed=True, facecolor='lightpink')
    ax.add_patch(blue_rectan)
    ax.add_patch(pink_rectan)
    # 遍历数据，根据标签放点到不同的正方形
    for i in range(len(labels)):
        if labels[i]==0:
            x = random.uniform(0,10)
            y = random.uniform(0, 10-x)
        elif labels[i] == 1:
            while True:
                x = random.uniform(0,10)
                y = random.uniform(0,10)
                if y >= -x + 10 and y <= 10 and x <= 10:
                    break
        if data[i] == 0:
            ax.plot(x, y, marker='o',markersize=2,color='blue')
        elif data[i] == 1:
            ax.plot(x, y, marker='o',markersize=2,color='red')

    # 设置坐标轴范围
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    # 隐藏坐标轴
    #ax1.axis('off')
    #ax2.axis('off')
    plt.savefig('sandiantu/{}.png'.format(name))
    #plt.show()

    plt.close()

with open('sandiantu/ucf_abn_bank.pickle', 'rb') as file:
    abn_index_all = pickle.load(file)
    file.close()
with open('sandiantu/ucf_label_bank.pickle', 'rb') as file:
    gt_all = pickle.load(file)
    file.close()
abn_index = np.zeros(0)
gt = np.zeros(0)
best_names_ucf = ["Arrest001_x264","Burglary018_x264","Explosion013_x264","RoadAccidents011_x264","RoadAccidents019_x264","RoadAccidents121_x264","RoadAccidents122_x264","RoadAccidents125_x264","RoadAccidents131_x264","RoadAccidents133_x264"]
# best_names_xd = ["Black.Hawk.Down.2001__#01-42-58_01-43-58_label_G-0-0",\
#                 "Bullet.in.the.Head.1990__#00-23-31_00-24-40_label_G-0-0","God.Bless.America.2011__#01-32-00_01-32-50_label_B2-0-0",\
#                 "GoldenEye.1995__#00-10-00_00-10-40_label_G-0-0","Ip.Man.3.2015__#00-17-51_00-18-51_label_B1-0-0","Ip.Man.2008__#01-14-31_01-15-24_label_B1-0-0",\
#                 "Jason.Bourne.2016__#0-50-20_0-50-30_label_G-0-0","Kingsman.The.Golden.Circle.2017__#00-41-22_00-41-32_label_B2-0-0","Kingsman.The.Secret.Service.2014__#00-22-10_00-23-10_label_B2-0-0",\
#                 "Love.Death.and.Robots.S01E10__#0-01-00_0-02-13_label_B2-G-0","Mission.Impossible.II.2000__#01-25-08_01-25-31_label_B2-0-0",\
#                 "Mission.Impossible.II.2000__#01-32-56_01-33-25_label_B1-0-0","Mission.Impossible.III.2006__#00-56-23_00-56-40_label_G-0-0","Rush.Hour.1998.BluRay__#01-25-21_01-25-46_label_B2-0-0",\
#                 "Spectre.2015__#02-15-30_02-16-05_label_G-0-0","Taken.2.UNRATED.EXTENDED.2012__#01-00-00_01-00-16_label_B2-0-0","Taken.3.2014__#01-09-26_01-09-54_label_G-0-0","Tropa.de.Elite.2.2010__#00-46-11_00-46-50_label_B2-0-0","v=9CWJd1SezkA__#1_label_G-0-0","v=cEOM18n8fhU__#1_label_G-0-0",\
#                 "v=DtxU8UYiFws__#1_label_G-0-0","v=DVdXoVUNkhg__#1_label_G-0-0","v=LJ0Pu5_Mefs__#1_label_G-0-0"]
for k in best_names_ucf:
    v_abn = abn_index_all[k]
    v_gt = gt_all[k]
    v_abn = v_abn[:len(v_gt)]
    v_gt = v_gt[:len(v_abn)]
    abn_index = np.concatenate((abn_index,v_abn))
    gt = np.concatenate((gt,v_gt))
    # abn_index = v_abn
    # gt = v_gt
    # abn_index = abn_index.reshape(-1, 16)
    # abn_index = np.sum(abn_index, axis=-1)
    # abn_index[abn_index > 0] = 1
    # gt = gt.reshape(-1, 16)
    # gt = np.sum(gt, axis=-1)
    # gt[gt > 0] = 1
    # draw_abn(abn_index, gt, k)

abn_index = abn_index.reshape(-1,16)
abn_index = np.sum(abn_index,axis=-1)
abn_index[ abn_index > 0 ] = 1
gt = gt.reshape(-1,16)
gt = np.sum(gt,axis=-1)
gt[ gt > 0 ] = 1
draw_abn(abn_index,gt,'ucf')


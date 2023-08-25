from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
from numpy import *
import argparse
import pandas as pd
ap = argparse.ArgumentParser(description='MRGNN testing for the DBLP dataset')
ap.add_argument('--dataset', type=str,default='ACM')
args = ap.parse_args()

# fig = plt.figure(figsize=(5, 5))
X_gat = np.load("embedding_{}_gat.npy".format(args.dataset))
color_gat=np.load("color_{}_gat.npy".format(args.dataset))
X_ours = np.load("embedding_{}.npy".format(args.dataset))
color_ours=np.load("color_{}.npy".format(args.dataset))
X_magnn = np.load("embedding_{}_magnn.npy".format(args.dataset))
color_magnn=np.load("color_{}_magnn.npy".format(args.dataset))
X_magnn_ours = np.load("embedding_{}_magnn_ours.npy".format(args.dataset))
color_magnn_ours=np.load("color_{}_magnn_ours.npy".format(args.dataset))
# print(color.shape)
# print(X.shape)
# for i in color_gat:
#     if i==0:
#         i=4
#     elif i==2:
#         i=3
# color_gat=[4 for i in color_gat if i==0]
# color_gat=[3 for i in color_gat if i==2]
# color_gat = plt.cm.Set2(color_gat+2)
# color_ours = plt.cm.Set2(color_ours+2)
# color_magnn = plt.cm.Set2(color_magnn+2)
# color_magnn_ours = plt.cm.Set2(color_magnn_ours+2)
# color=str(color)
# color=color.replace("0","#6b5acf")
# color=color.replace("1","#6b5acf")
# color=color.replace("2","#6b5acf")
# color=color.replace("3","#6b5acf")
# colors=["#800000","#6b5acf","#fafaf4","#860102"]
fig,ax=plt.subplots(1,4,figsize=(40, 10))


tsne = TSNE(n_components=2,init='pca',n_iter=3000,learning_rate=250)

Y_ours = tsne.fit_transform(X_ours)
x_min, x_max = Y_ours.min(0), Y_ours.max(0)
Y_ours_finall = (Y_ours - x_min) / (x_max - x_min)
ax[0].scatter(Y_ours_finall[:, 0], Y_ours_finall[:, 1], c=color_ours, cmap=plt.cm.Spectral,s=4)
# ax[0,0].set_title("t-SNE: {}_ours".format(args.dataset),y=-0.1)
ax[0].set_title("GCANet(Ours)".format(args.dataset),y=-0.1)
ax[0].yaxis.set_ticks([])
ax[0].xaxis.set_ticks([])
ax[0].axis('off')

Y_gat = tsne.fit_transform(X_gat)
x_min, x_max = Y_gat.min(0), Y_gat.max(0)
Y_gat_finall = (Y_gat - x_min) / (x_max - x_min)
ax[1].scatter(Y_gat_finall[:, 0], Y_gat_finall[:, 1], c=color_gat, cmap=plt.cm.Spectral,s=4)
# ax[0,1].set_title("t-SNE: {}_gat".format(args.dataset),y=-0.1)
ax[1].set_title("GAT".format(args.dataset),y=-0.1)
ax[1].yaxis.set_ticks([])
ax[1].xaxis.set_ticks([])
ax[1].axis('off')

Y_magnn = tsne.fit_transform(X_magnn)
x_min, x_max = Y_magnn.min(0), Y_magnn.max(0)
Y_magnn_finall = (Y_magnn - x_min) / (x_max - x_min)
ax[2].scatter(Y_magnn_finall[:, 0], Y_magnn_finall[:, 1], c=color_magnn, cmap=plt.cm.Spectral,s=4)
# ax[1,0].set_title("t-SNE: {}_magnn".format(args.dataset),y=-0.1)
ax[2].set_title("MAGNN".format(args.dataset),y=-0.1)
ax[2].yaxis.set_ticks([])
ax[2].xaxis.set_ticks([])
ax[2].axis('off')

Y_magnn_ours = tsne.fit_transform(X_magnn_ours)
x_min, x_max = Y_magnn_ours.min(0), Y_magnn_ours.max(0)
Y_magnn_ours_finall = (Y_magnn_ours - x_min) / (x_max - x_min)
# ax[1,1].scatter(Y_magnn_ours_finall[:, 0], Y_magnn_ours_finall[:, 1], c=color_magnn_ours, cmap=plt.cm.Spectral,s=4)
ax[3].scatter(Y_magnn_ours_finall[:, 0], Y_magnn_ours_finall[:, 1], c=color_magnn_ours, cmap=plt.cm.Spectral,s=4)
# ax[1,1].set_title("t-SNE: {}_magnn_ours".format(args.dataset),y=-0.1)
ax[3].set_title("MAGNN_GCA".format(args.dataset),y=-0.1)
ax[3].yaxis.set_ticks([])
ax[3].xaxis.set_ticks([])
ax[3].axis('off')

# print(color)

# ax = fig.add_subplot(2, 1, 2)
# for i in range(4):
#     temp=np.where(color==i)[0]
#     # print(temp)
#     plt.scatter(Y[temp, 0], Y[temp, 1], c=colors[i],s=4, cmap=plt.cm.Spectral)


# ax.xaxis.set_major_formatter(NullFormatter())  # 设置标签显示格式为空
# ax.yaxis.set_major_formatter(NullFormatter())
# plt.axis('tight')
# ax.axes.get_yaxis().set_visible(False)
# x 轴不可见
# ax.axes.get_xaxis().set_visible(False)

plt.savefig("{}_allHEN.png".format(args.dataset))









# micro=[0.6678648454349388, 0.660531885524984, 0.6686605038970456, 0.6748786189534257]
# macro=[0.6368117727719028, 0.6104679065018731, 0.6297434343684811, 0.6442724505145657]
# print('ave_micro:{} std_micro:{}'.format(mean(micro),std(micro)))
# print('ave_macro:{} std_macro:{}'.format(mean(macro),std(macro)))
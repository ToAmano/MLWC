# 無地のmatplotlibのグラフの作成
# * D論用の図表（tikzあり）
import numpy as np
# load_dir="merge_20230322/model2/"
load_dir="./"


# ch
ch_test_pred =np.load(load_dir+"ch_model/ch_valid_pred.npy")
ch_train_pred=np.load(load_dir+"ch_model/ch_train_pred.npy")
ch_test_true =np.load(load_dir+"ch_model/ch_valid_true.npy")
ch_train_true=np.load(load_dir+"ch_model/ch_train_true.npy")
# co
co_test_pred =np.load(load_dir+"co_model/co_valid_pred.npy")
co_train_pred=np.load(load_dir+"co_model/co_train_pred.npy")
co_test_true =np.load(load_dir+"co_model/co_valid_true.npy")
co_train_true=np.load(load_dir+"co_model/co_train_true.npy")
# oh
oh_test_pred =np.load(load_dir+"oh_model/oh_valid_pred.npy")
oh_train_pred=np.load(load_dir+"oh_model/oh_train_pred.npy")
oh_test_true =np.load(load_dir+"oh_model/oh_valid_true.npy")
oh_train_true=np.load(load_dir+"oh_model/oh_train_true.npy")
# o
o_test_pred =np.load(load_dir+"o_model/o_valid_pred.npy")
o_train_pred=np.load(load_dir+"o_model/o_train_pred.npy")
o_test_true =np.load(load_dir+"o_model/o_valid_true.npy")
o_train_true=np.load(load_dir+"o_model/o_train_true.npy")
# cc
cc_test_pred =np.load(load_dir+"cc_model/cc_valid_pred.npy")
cc_train_pred=np.load(load_dir+"cc_model/cc_train_pred.npy")
cc_test_true =np.load(load_dir+"cc_model/cc_valid_true.npy")
cc_train_true=np.load(load_dir+"cc_model/cc_train_true.npy")

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8.6,6.45),tight_layout=True) # figure, axesオブジェクトを作成

# color map
# https://colorbrewer2.org/#type=qualitative&scheme=Set1&n=5
orange="#ff7f00"
blue="#377eb8"
green="#4daf4a"
red="#e41a1c"
purple="#984ea3"

# 描画
scatter1=ax.scatter(np.linalg.norm(co_train_pred,axis=1), np.linalg.norm(co_train_true,axis=1), label="train",alpha=0.2,color=orange, s=5)  
scatter2=ax.scatter(np.linalg.norm(co_test_pred,axis=1),  np.linalg.norm(co_test_true,axis=1),label="test",alpha=0.2, color=orange,s=5)
# ax.text(0.33, 0.5, 'co', transform=ax.transAxes, fontsize=20, verticalalignment='top') #

ax.scatter(np.linalg.norm(cc_train_pred,axis=1), np.linalg.norm(cc_train_true,axis=1),alpha=0.1,color=purple,s=5)
ax.scatter(np.linalg.norm(cc_test_pred,axis=1),  np.linalg.norm(cc_test_true,axis=1),alpha=0.1, color=purple,s=5)

ax.scatter(np.linalg.norm(ch_train_pred,axis=1),np.linalg.norm(ch_train_true,axis=1), alpha=0.1,color=blue,s=5)
ax.scatter(np.linalg.norm(ch_test_pred,axis=1), np.linalg.norm(ch_test_true,axis=1),alpha=0.1, color=blue,s=5)
# ax.text(0.43, 0.6, 'ch', transform=ax.transAxes, fontsize=20, verticalalignment='top') #

ax.scatter(np.linalg.norm(oh_train_pred,axis=1),np.linalg.norm(oh_train_true,axis=1),alpha=0.1,color=green,s=5)
ax.scatter(np.linalg.norm(oh_test_pred,axis=1), np.linalg.norm(oh_test_true,axis=1),alpha=0.1, color=green,s=5)
# ax.text(0.1, 0.25, 'oh', transform=ax.transAxes, fontsize=20, verticalalignment='top') #

ax.scatter(np.linalg.norm(o_train_pred,axis=1), np.linalg.norm(o_train_true,axis=1),alpha=0.1,color=red,s=5)
ax.scatter(np.linalg.norm(o_test_pred,axis=1),  np.linalg.norm(o_test_true,axis=1),alpha=0.1, color=red,s=5)
# ax.text(0.65, 0.8, 'o', transform=ax.transAxes, fontsize=20, verticalalignment='top') #


# ax.text(0.03, 0.18, 'cc', transform=ax.transAxes, fontsize=20, verticalalignment='top') #


## ax.plot(kayser_exp*33.3, diel_exp, label="Exp. [5]", lw=3)  # 描画

# ax.scatter(cc_train_pred,cc_train_true,alpha=0.1,color="#1f77b4",s=5)
# ax.scatter(cc_test_pred,cc_test_true,alpha=0.1, color='#ff7f0e',s=5)

ax.set_xlim(0,4)
ax.set_ylim(0,4)

ax.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
# https://qiita.com/nyunyu122/items/662c4b69b7b165c4b767
ax.axis('tight')
ax.axis('off')
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
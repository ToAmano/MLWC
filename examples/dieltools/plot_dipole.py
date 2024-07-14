import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
# load data
dnn = np.loadtxt("dipole_10ps/total_dipole.txt")[:,1:]
# load timestep
with open('dipole_10ps/total_dipole.txt')as f:
    line= f.readline()
    print(line)
    while line:
        if line.startswith("#TIMESTEP") == True:
            dt = line.split(" ")[1]  # timestep in fs
            print(dt)
            break
# constant change from fs to ps
fs2ps = 1/1000
# figure instantce
fig, ax = plt.subplots(figsize=(8,5),tight_layout=True) 
ax.plot(dt*fs2ps*np.arange(len(dnn[:,0])), dnn[:,0], label="DNN_x", lw=3)  # 描画
ax.plot(dt*fs2ps*np.arange(len(dnn[:,1])), dnn[:,1], label="DNN_y", lw=3)  # 描画
ax.plot(dt*fs2ps*np.arange(len(dnn[:,2])), dnn[:,2], label="DNN_z", lw=3)  # 描画

# 各要素で設定したい文字列の取得
xticklabels = ax.get_xticklabels()
yticklabels = ax.get_yticklabels()
xlabel="Time [ps]"
ylabel="Dipole [D]"

# 各要素の設定を行うsetコマンド
ax.set_xlabel(xlabel,fontsize=22)
ax.set_ylabel(ylabel,fontsize=22)
ax.grid()
ax.tick_params(axis='x', labelsize=20 )
ax.tick_params(axis='y', labelsize=20 )
lgnd=ax.legend(loc="upper left",fontsize=20)
# lgnd.legendHandles[0]._sizes = [30]
# lgnd.legendHandles[0]._alpha = [1.0]
fig.savefig("test.png")

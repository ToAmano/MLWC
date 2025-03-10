import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# load data
df = pd.read_csv("dipole_10ps/total_dipole.txt_diel.csv")

# figure instantce
fig, ax = plt.subplots(figsize=(8,5),tight_layout=True) 
ax.plot(df["freq_kayser"], df["imag_diel"],label="imag_diel",lw=3)
ax.set_xlim(0,3500)
# 各要素で設定したい文字列の取得
xticklabels = ax.get_xticklabels()
yticklabels = ax.get_yticklabels()
xlabel="Frequency [cm-1]"
ylabel="Epsilon"

# 各要素の設定を行うsetコマンド
ax.set_xlabel(xlabel,fontsize=22)
ax.set_ylabel(ylabel,fontsize=22)
ax.grid()
ax.tick_params(axis='x', labelsize=20 )
ax.tick_params(axis='y', labelsize=20 )
lgnd=ax.legend(loc="upper left",fontsize=20)
# lgnd.legendHandles[0]._sizes = [30]
# lgnd.legendHandles[0]._alpha = [1.0]
fig.savefig("imag_diel.png")

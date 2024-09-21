import matplotlib.pyplot as plt
# plt.hist(hydrogen_bond_list.flatten(),bins=100)# 
from ase.geometry import get_distances
import pandas as pd
import ase
import numpy as np



def calc_roo(traj_liquid:list[ase.Atoms],oxygen_list:list[int],NUM_MOL:int,NUM_ATOM_ALL:int)->np.ndarray:
    
    # Create an array of molecule IDs, each repeated 36 times
    molecule_ids = np.repeat(np.arange(NUM_MOL), NUM_ATOM_ALL)
    # extract oxygens
    oxygen_molecule_ids = molecule_ids[oxygen_list]

    # define mask
    n = len(oxygen_list)
    mask = np.ones((n, n), dtype=bool)  # Start with all pairs being valid (True)

    for i in range(NUM_MOL):    
        idx = np.where(oxygen_molecule_ids == i)[0] # [0] is necessary to get the indices
        for idx_1 in idx:
            for idx_2 in idx:
                # print(idx_1, idx_2)
                mask[idx_1,idx_2] = False
    
    roo_list = np.zeros((len(traj_liquid),len(oxygen_list)))
    for counter,atoms in enumerate(traj_liquid): # frameに関するloop
        pos = atoms.get_positions()
        distances, distances_len = get_distances(pos[oxygen_list],pos[oxygen_list],cell=atoms.get_cell(),pbc=True)
        #distances = np.array(distances)
        # print(distances_len)
        #print(np.shape(distances_len))
        #print(distances)
        #print(np.shape(distances))
        # Apply the mask to filter out invalid distances
        valid_distances = np.where(mask, distances_len, np.inf)  # Set intra-molecular distances to infinity
    
        # !! We only take the 1st closest oxygen atom. This could be potentially wrong for complex systems.
        hb_length = np.sort(valid_distances,axis=1)[:,0] # 分子外で最も近いものをとってくる．
        # print(np.shape(hb_length))
        roo_list[counter] = hb_length
    return roo_list


def make_df_acf(acf:np.ndarray,timestep_fs:float) -> pd.DataFrame:
    """make DataFrame from acf

    Args:
        acf (np.ndarray): acf
        timestep (float): timestep in fs

    Returns:
        pd.DataFrame: acf
    """
    import pandas as pd
    # pandas化
    df = pd.DataFrame()
    df["time_fs"] = np.arange(len(acf))*timestep_fs # fs
    df["acf"] = acf
    df["acf_normalized"] = df["acf"]/df["acf"][0] # normalize
    return df

def calc_lengthcorr(acf:np.ndarray, timestep_fs:float)->pd.DataFrame:
    """calculate FT from acf for vdos

    Args:
        acf (np.ndarray): _description_
        timestep (float): timestep in fs

    Returns:
        pd.DataFrame: _description_
    """
    import numpy as np
    if len(np.shape(acf)) != 1: # check acf is 1d
        raise ValueError("ERROR :: acf shape not correct")    
    TIMESTEP = timestep_fs/1000 # fs to ps
    # logger.info("TIMESTEP [ps] :: {0}".format(TIMESTEP))

    time_data=len(acf) # データの長さ
    freq=np.fft.fftfreq(time_data, d=TIMESTEP) # omega
    length=freq.shape[0]//2 + 1 # rfftでは，fftfreqのうちの半分しか使わない．
    rfreq=freq[0:length] # これが振動数(in THz)

    #usage:: numpy.fft.fft(data, n=None, axis=-1, norm=None)
    ans=np.fft.rfft(acf, norm="forward") #こっちが1/Nがかかる規格化．(time_data?)
    #ans=np.fft.rfft(fft_data, norm="backward") #その他の規格化1:何もかからない
    #ans=np.fft.rfft(fft_data, norm="ortho")　　#その他の規格化2:1/sqrt(N))がかかる

    ans_real_denoise= ans.real-ans.real[-1] # 振幅が閾値未満はゼロにする（ノイズ除去）
    # print(ans.real)
    ans = ans_real_denoise + ans.imag*1j # 再度定義のし直しが必要

    # VDOS:: time_data*TIMESTEPは合計時間をかける意味
    fftreal = 2*ans.real*(time_data*TIMESTEP) 
    
    # pandas化
    import pandas as pd
    df = pd.DataFrame()
    df["thz"] = rfreq
    df["freq_kayser"] = rfreq*33.3
    df["roo"] = fftreal # -fftvdos[0] # subtract vdos(omega=0) to assure vdos(0)=0
    return df


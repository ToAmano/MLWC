'''
ase.atomsのリストから双極子を計算する．
まだコードは実験的であり，最終的にはcustom_trajクラスのメソッドとして実装することを目指す．
'''

import sys
import numpy as np
import matplotlib.pyplot as plt
import ase.units
import statsmodels.api as sm 
    

class atomic_charge():
    '''
    wfcの計算で使う用に主要な原子の原子電荷を定義する．
    '''
    charges = {
        'He' : -2, # Heをワニエセンターとして扱っている．
        "H"  :  1,
        "C"  :  4,
        "O"  :  6,
        "Si" :  4
    }


class dipole():
    '''
    dipoleを操作するためのクラス．dipoleに加えて誘電関数計算に必要な時間情報，結晶構造の情報を保持する．
    '''
    def __init__(self, dipole, time, unitcell_vector):
        self.dipole          = dipole
        self.time            = time
        self.UNITCELL_VECTOR = unitcell_vector

    def plot_dipole(self, start:int=0, stop:int=-1):
        return raw_plot_dipole(self.dipole, self.time, start, stop)
        
    def calc_acf(self, start:int=0, stop:int=-1, Temp=300):
        return 0
    

    
def get_charges(atoms_list):
    '''
    in case of wannier :: set atomic charge

    Notes
    ----------------
    汎関数によってもどこまでを電子として扱うかが異なるため定義が一意ではないところが少し問題．．．
    この問題を一時的に回避するため，atomsにchargeを追加する専用の関数を定義しておく．
    こうすれば電荷のカスタムに対応する．最終的にはWCの数と対応するかをチェックする．
    '''
    charge_list=[]
    for i in atoms_list:
        charge=[]
        for j in i.get_chemical_symbols():
            charge.append(atomic_charge.charges[j])
        charge_list.append(charge)

    # 電荷の総和が0になっているかの確認
    for i in charge_list:
        if not np.abs(np.sum(np.array(i))) < 1.0e-5:
            print("ERRIR :: total charge is not zero :: ", np.sum(np.array(i)))
            sys.exit()
            
    return charge_list
        

def add_charges(atoms_list,charge_list):
    '''
    電荷のリストを与えるとそれをase.atomsのリストに自動で加えてくれる．
    加えた電荷はase.get.charges()で確認できる．
    '''
    # 長さが等しいかのテスト
    if not len(atoms_list) == len(charge_list):
        print("ERROR :: steps of 2 files differ")
        print("steps for atoms_list :: ", len(atoms_list))
        print("steps for charge_list :: ", len(charge_list))
    if not len(atoms_list[0].get_chemical_symbols()) == len(charge_list[0]):
        print("ERROR :: # of atoms differ")
        print("# of atoms for atoms_list :: ", len(atoms_list[0].get_chemical_symbols()))
        print("# of atoms for charge_list :: ", len(charge_list[0]))
        
    
    for i in range(len(atoms_list)):
        atoms_list[i].set_initial_charges(charge_list[i])

    return atoms_list
    

def calc_dipoles(atoms_list):
    '''
    calculate dipole in Debye units. atoms_list must include charges.

    Notes
    ----------------
    aseでは長さがAngstrom,電荷は電子電荷で扱っており，一方でDebyeは
    3.33564×10−30 C·mで定義されているので，これを変換している．

    Debye   = 3.33564e-30
    charge  = 1.602176634e-19
    ang      = 1.0e-10 
    coef    = ang*charge/Debye
    print(coef)
   
    基本的には
    1[Ang*e]=4.8032[Debye]となる．
    '''
    
    dipole_array=[]

    for i in range(len(atoms_list)):
        tmp_dipole=np.einsum("i,ij -> j",atoms_list[i].get_initial_charges(),atoms_list[i].get_positions())
        dipole_array.append(tmp_dipole)
    #
    dipole_array=np.array(dipole_array)/ase.units.Debye
    return dipole_array



def raw_plot_dipole(dipole_array, time, start:int=0, stop:int=-1):
    '''
    dipoleの経時変化をmatplotlibでプロットする.

    input
    ---------------
    dipole_array : n*3 numpy array
        input dipole moment in [D]

    time         : n numpy array
        input time series

    start        : start step
        start step
    '''
    if start > np.shape(dipole_array)[0]: 
        print("ERROR :: start step is larger than dipole_array size")
    if stop > 0 and stop > np.shape(dipole_array)[0] :
        print("ERROR :: stop step is larger than dipole_array size")


    plt.plot(time[start:stop],dipole_array[start:stop,0]-dipole_array[start,0],label="Dipole_x")
    plt.plot(time[start:stop],dipole_array[start:stop,1]-dipole_array[start,1],label="Dipole_y")
    plt.plot(time[start:stop],dipole_array[start:stop,2]-dipole_array[start,2],label="Dipole_z")
    plt.xlabel("timestep")
    plt.ylabel("Dipole [D]")
    plt.legend()
    return plt



def raw_calc_acf(dipole, unitcell_vector, start=0, stop=-1, T=300):
    '''
    dipole :: D
    time   :: ps
    unitcell_vector :: A
    T      :: K

    output
    -----------
    eps_0 ::
    acf_i :: Debye
    
    '''
    eps0  = ase.units._eps0    #8.8541878128e-12
    debye = 1/ase.units._c*1e-21 #1/ase.units.Debye #3.33564e-30 
    A3    = 1/ase.units.m/ase.units.m/ase.units.m #m^3 
    kb    = ase.units._k         #1.38064852e-23

    # 
    kbT = kb * T 
    V=  np.dot( unitcell_vector[0], np.cross(unitcell_vector[1], unitcell_vector[2])) * A3 # 11.1923*11.1923*11.1923 * A3
    print(" -------------- ")
    print(" volume :: ", V)
    print("")
    # N = int(len(ms))

    # cut sart:stop
    dipole=dipole[start:stop,:]
    
    N=int(np.shape(dipole)[0])
    print(" -------------- ")
    print(" nlag   :: ", N)
    
    # 各軸のdipoleを抽出．平均値を引く(eps_0のため．ACFは実装上影響なし)
    dMx=dipole[:,0]-np.mean(dipole[:,0])
    dMy=dipole[:,1]-np.mean(dipole[:,1])
    dMz=dipole[:,2]-np.mean(dipole[:,2])
    
    eps_0 = 1.0 + ((np.mean(dMx**2+dMy**2+dMz**2))*debye**2)/(3.0*V*kbT*eps0)
    
    #自己相関関数を求める (基本的にはfftはTrueで問題ない．)
    acf_x = sm.tsa.stattools.acf(dMx,nlags=N,fft=True)
    acf_y = sm.tsa.stattools.acf(dMy,nlags=N,fft=True)
    acf_z = sm.tsa.stattools.acf(dMz,nlags=N,fft=True)
    pred_data =(acf_x+acf_y+acf_z)/3
    return pred_data # eps_0, acf_x, acf_y, acf_z
    


def calc_nonnormalized_acf(dipole, unitcell_vector, start=0, stop=-1, T=300):
    '''
    dipole :: D
    time   :: ps
    unitcell_vector :: A
    T      :: K

    output
    -----------
    eps_0 ::
    acf_i :: Debye
    
    '''
    eps0  = ase.units._eps0    #8.8541878128e-12
    debye = 1/ase.units._c*1e-21 #1/ase.units.Debye #3.33564e-30 
    A3    = 1/ase.units.m/ase.units.m/ase.units.m #m^3 
    kb    = ase.units._k         #1.38064852e-23

    # 
    kbT = kb * T 
    V=  np.dot( unitcell_vector[0], np.cross(unitcell_vector[1], unitcell_vector[2])) * A3 # 11.1923*11.1923*11.1923 * A3
    print(" -------------- ")
    print(" volume :: ", V)
    print("")
    # N = int(len(ms))

    # cut sart:stop
    dipole=dipole[start:stop,:]
    
    N=int(np.shape(dipole)[0]/2)
    print(" -------------- ")
    print(" nlag   :: ", N)
    
    # 各軸のdipoleを抽出．平均値を引く(eps_0のため．ACFは実装上影響なし)
    dMx=dipole[:,0]-np.mean(dipole[:,0])
    dMy=dipole[:,1]-np.mean(dipole[:,1])
    dMz=dipole[:,2]-np.mean(dipole[:,2])

    #自己相関関数を求める（acf(t=0)でnormalizeしない．）
    acf_x = sm.tsa.stattools.acf(dMx,nlags=N,fft=False) * np.std(dMx) * np.std(dMx)
    acf_y = sm.tsa.stattools.acf(dMy,nlags=N,fft=False) * np.std(dMx) * np.std(dMx)
    acf_z = sm.tsa.stattools.acf(dMz,nlags=N,fft=False) * np.std(dMx) * np.std(dMx)
    pred_data =(acf_x+acf_y+acf_z)/3
    return pred_data


def calc_eps0(dipole, unitcell_vector, start=0, stop=-1, T=300):
    '''
    誘電定数を計算
    '''
    '''
    dipole :: D
    time   :: ps
    unitcell_vector :: A
    T      :: K

    output
    -----------
    eps_0 ::
    acf_i :: Debye
    
    '''
    eps0  = ase.units._eps0    #8.8541878128e-12
    debye = 1/ase.units._c*1e-21 #1/ase.units.Debye #3.33564e-30 
    A3    = 1/ase.units.m/ase.units.m/ase.units.m #m^3 
    kb    = ase.units._k         #1.38064852e-23

    # 
    kbT = kb * T 
    V=  np.dot( unitcell_vector[0], np.cross(unitcell_vector[1], unitcell_vector[2])) * A3 # 11.1923*11.1923*11.1923 * A3
    print(" -------------- ")
    print(" volume :: ", V)
    print("")
    # N = int(len(ms))

    # cut sart:stop
    dipole=dipole[start:stop,:]
        
    # 各軸のdipoleを抽出．平均値を引く(eps_0のため．ACFは実装上影響なし)
    dMx=dipole[:,0]-np.mean(dipole[:,0])
    dMy=dipole[:,1]-np.mean(dipole[:,1])
    dMz=dipole[:,2]-np.mean(dipole[:,2])
    # 実際のeps0の計算    
    eps_0 = 1.0 + ((np.mean(dMx**2+dMy**2+dMz**2))*debye**2)/(3.0*V*kbT*eps0)

    return eps_0


def calc_aveM2(dipole, start=0, stop=-1):
    '''
    <M^2>を計算する．もしもtrajectoryが十分長ければ一定値に収束するはず．
    '''
    '''
    dipole :: 単位は[D]
    time   :: ps
    unitcell_vector :: A
    T      :: K

    output
    -----------
    eps_0 ::
    acf_i :: Debye
    
    '''

    # cut sart:stop
    dipole=dipole[start:stop,:]
        
    # 各軸のdipoleを抽出．平均値を引く(eps_0のため．ACFは実装上影響なし)
    dMx=dipole[:,0]
    dMy=dipole[:,1]
    dMz=dipole[:,2]

    # 平均値計算
    mean_M2=(np.mean(dMx**2)+np.mean(dMy**2)+np.mean(dMz**2))
    return mean_M2


def calc_aveM(dipole, start=0, stop=-1):
    '''
    <M>^2 (スカラー）を計算する．もしもtrajectoryが十分長ければ0になるはず．
    '''
    '''
    dipole :: 単位は[D]
    
    output
    -----------
    eps_0 ::
    acf_i :: Debye
    
    '''

    # cut sart:stop
    dipole=dipole[start:stop,:]
        
    # 各軸のdipoleを抽出．平均値を引く(eps_0のため．ACFは実装上影響なし)
    dMx=dipole[:,0]
    dMy=dipole[:,1]
    dMz=dipole[:,2]

    # 平均値計算
    mean_M=np.mean(dMx)**2+np.mean(dMy)**2+np.mean(dMz)**2
    
    return mean_M




def plot_ACF(acf_x, time):
    '''
    自己相関関数の図示
    '''
    plt.plot(acf_x,label="acf")
    plt.legend()
    plt.xlabel("timestep")
    plt.ylabel("ACF")
    plt.title("ACF vs timestep")
    return plt


class acf():
    # 自己相関のデータのみを保存する．
    def __init__(self,time,acf):
        import pandas as pd
        self.acf_df = pd.DataFrame()
        self.acf_df["time"] = time # ps
        self.acf_df["acf"]  = acf  # normalized acf

class diel_function():
    '''
    誘電関数のクラス．主にrefractive indexやalphaを計算するために利用する．
    TODO :: kayserとはあるが，THz単位でinputする
    '''
    def __init__(self, kayser, ffteps1, ffteps2, step:int=1):
        """_summary_

        Args:
            kayser (_type_): _description_
            ffteps1 (_type_): _description_
            ffteps2 (_type_): _description_
            step (int): step to moving average (default=1, no-average)
        """
        import pandas as pd
        if step < 1:
            print("ERROR :: step must be larger than 1")
        self.step:int = step
        self.diel_df = pd.DataFrame()
        self.diel_df["freq_thz"] = kayser
        self.diel_df["freq_kayser"] = kayser*33.3
        self.diel_df["real_diel"]   = ffteps1
        self.diel_df["imag_diel"]   = ffteps2
        self.diel_df["alphan"]      = raw_calculate_absorption(self.diel_df) # alpha(omega)n(omega)の計算
        # 2024/3/22 apply moving average to imag_diel]
        window = np.ones(self.step)/self.step 
        self.diel_df["imag_diel"]   = np.convolve(ffteps2,window,mode="same")
        print("The DataFrame generated from the NumPy array is:")
        print(self.diel_df)
        # refractive_index&alphaを計算してpandasに格納
        self.refractive_df = self.calc_refractiveindex()
        
        
        
    def calc_refractiveindex(self):
        """calculate refractive index from dielectric functions

        Returns:
            _type_: _description_
        """
        return raw_calculate_refractiveindex_pandas(self.diel_df,self.step)
    
    def calc_alpha(self):
        '''
        alphaの計算式の出典
        alphaは，2omega*kappa/cとなる．通常，横軸はomegaではなくf = omega/2piとなる．(freq_kayserはfである．)
        ここで，freq_kayserはomegaではなくfであることに注意が必要．
        従って，計算手順としては
         1:omega = refractive_index["freq_kayser"]/2pi [cm-1]を計算
         2:単位をTHzに変換 omega -> omega/33.3
         3: kappaは無次元量なのでそのまま利用する．(誘電関数を無次元とした場合)
         4: 光速c=299 792 458 m/s ~ 3.0e8を代入する．
         5: 光速をcm*THzに変換する．1THz =10^12Hz = 10^12/s より， 3e8[m/s]= 3e-2 [cm*THz]
         5: 単位を調整する．kappa * THz/(m/s) = 1e-10/3e8
         
        注意！！誘電関数と複素屈折率の関係は，非誘電関数との関係として定義されており，複素屈折率は無次元．
        '''
        refractive_index = self.calc_refractiveindex()
        window = np.ones(self.step)/self.step 
        return np.convolve(refractive_index["imag_ref_index"]*refractive_index["freq_kayser"]/33.3*400*3.14/3,window,mode="same")
    
    def save_dielec(self,filename:str):
        """save dielectric function dataframe to csv

        Args:
            filename (str): filename to save 

        Returns:
            _type_: _description_
        """
        import os
        if os.path.isfile(filename):
            print("ERROR file exists !!")
            return 0
        self.diel_df.to_csv(filename)
        return 0



def raw_calculate_refractiveindex(kayser, ffteps1, ffteps2,step:int=1):
    """kayser, ffteps1, ffteps2からrefractive indexを計算する．

    Args:
        kayser (_type_): _description_
        ffteps1 (_type_): _description_
        ffteps2 (_type_): _description_
        step (int, optional): step to moving-average. Defaults to 1.

    Returns:
        _type_: _description_
    """

    import pandas as pd
    import cmath
    epsilon= ffteps1+1j*ffteps2 #本来はここはマイナスだが，プラスで計算しておくと（kappaもマイナスで定義されているので）全体として辻褄が合うようになっている．
    refractive_index=[]
    re_refractive_index=[]
    im_refractive_index=[]

    for i in epsilon:
        a,b = cmath.polar(i)
        refractive_index.append(cmath.rect(np.sqrt(a),b/2))

    re_refractive_index = [a.real for a in refractive_index ] 
    im_refractive_index = [a.imag for a in refractive_index ]  
    
    data_df = pd.DataFrame()
    print("The DataFrame generated from the NumPy array is:")
    data_df["freq_kayser"] = kayser
    data_df["real_ref_index"]   = re_refractive_index
    data_df["imag_ref_index"]   = im_refractive_index    
    data_df["alpha"] = raw_calculate_alpha(data_df,step) # alphaも計算
    # 最後にalphaの計算
    return data_df


def raw_calculate_refractiveindex_pandas(eps_df,step:int=1):
    """dfから直接計算する方法

    Args:
        eps_df (_type_): _description_
        step (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    if step < 1:
        print("ERROR :: step must be larger than 1")
        return 0
    return raw_calculate_refractiveindex(eps_df["freq_kayser"], eps_df["real_diel"], eps_df["imag_diel"],step)

def raw_calculate_alpha(refractive_df,step:int=1):
    """Calculate alpha

    Args:
        refractive_df (_type_): _description_
        step (int, optional): step to moving-average. Defaults to 1.

    Returns:
        _type_: _description_
    """
    if step < 1:
        print("ERROR :: step must be larger than 1")
        return 0
    window = np.ones(step)/step # apply moving average
    return np.convolve(refractive_df["imag_ref_index"]*refractive_df["freq_kayser"]/33.3*400*3.14/3, window, mode="same")

def raw_calculate_absorption(df:pd.DataFrame):
    """Calculate absorption (alpha*n)

    Args:
        df (pd.DataFrame): dataframe of dielectric function

    Returns:
        _type_: _description_
    """
    # alpha(omega)n(omega)の計算
    return df["imag_diel"]*df["freq_kayser"]/33.3*200*3.14/3
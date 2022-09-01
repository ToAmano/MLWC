'''
ase.atomsのリストから双極子を計算する．
まだコードは実験的であり，最終的にはcustom_trajクラスのメソッドとして実装することを目指す．
'''

import sys
import numpy as np
import matplotlib.pyplot as plt
import ase.units
    

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


def plot_dipoles(dipole_array, start:int=-1, stop:int=-1):
    '''
    dipoleの経時変化をmatplotlibでプロットする

    input
    ---------------
    dipole_array : n*3 numpy array
        input dipole moment in [D]

    start        : start step
        start step
    '''
    if start > dipole_array.shape()[0]:
        print("ERROR :: start step is larger than dipole_array size")

        
    x=np.arange(dipole_array.shape()[0]-start)
    plt.plot(x,dipole_array[:,0]-dipole_array[:,0],label="Dipole_x")
    plt.plot(x,dipole_array[:,1]-dipole_array[:,1],label="Dipole_y")
    plt.plot(x,dipole_array[:,2]-dipole_array[:,2],label="Dipole_z")
    plt.xlabel("timestep")
    plt.ylabel("Dipole [D]")
    plt.legend()
    return plt

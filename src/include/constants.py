

class constant():
    '''
    計算で用いる物理定数を定めている．
    cpmd,ml,quadrupoleで利用するものを多く収録．
    命名規則も考える必要がある．

     - 先頭は大文字にする．
     - 通常の単位で与えた場合，その定数をSI単位で表したものとする．
     - それ以外への単位の変換（Bohr_to_Angなど）はA_to_Bの形とし，Aの単位をBの単位で表した時の係数とする．
    '''
    Debye   = 3.33564e-30      # Debye in Cm
    Charge  = 1.602176634e-019 # electron charge in C
    Ang      = 1.0e-10         # Ang in m

    Ang_to_Bohr = 1.8897259886
    # Bohr        =	5.29177210903E−11

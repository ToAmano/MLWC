
import numpy as np

def calc_bondcenter(vectors_array, bond_list):
    # check the shape of vectors_array
    if vectors_array.ndim != 3 or vectors_array.shape[2] != 3:
        raise ValueError(f"Invalid shape for vectors_array. Expected shape [a, b, 3], but got {np.shape(vectors_array)}.")
    
    # `bond_list` を numpy 配列に変換して、インデックス操作を効率化
    bond_list = np.array(bond_list)
    
    # ボンドの開始点と終了点を抽出
    start_points = vectors_array[:, bond_list[:, 0], :]
    end_points =   vectors_array[:, bond_list[:, 1], :]
    
    # ボンドの中点を計算
    bond_centers = (start_points + end_points) / 2.0
    
    return bond_centers
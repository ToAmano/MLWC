import numpy as np
import torch
class DataSet_custom():
    '''
    https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    入力としてnumpy arrayを受け取る想定で作成してみよう．
    '''
    def __init__(self,descs_x:np.ndarray,true_y:np.ndarray):
        import numpy as np
        import torch
        # convert from numpy to torch
        descs_x = torch.from_numpy(descs_x.astype(np.float32)).clone()
        true_y  = torch.from_numpy(true_y.astype(np.float32)).clone()
        self.x =  descs_x     # 入力
        self.y =  true_y     # 出力
        
    def __len__(self):
        return len(self.x) # データ数を返す
        
    def __getitem__(self, index):
        # index番目の入出力ペアを返す
        return self.x[index], self.y[index]

#!/usr/bin/env python3
# coding: utf-8

def find_input(inputs, str):
    '''
    入力ファイルから特定のキーワードをサーチする．
    
    input
    -------------
      inputs :: [keyword, value]がappendされた2次元配列．keywordを検索し，valueを返す
      
    output
    -------------
      output  :: keywordに対応するvalue.
      
    note 
    -------------
     TODO :: キーワードが複数出てきた時は？
     TODO :: optional keywordがこのままだと扱えない．
    '''
    output = None
    for i in inputs:
        if i[0] == str:
            output=i[1]
            print(" {0} :: {1}".format(str,output))
    if output == None:
        print(" ERROR :: input not found :: {}".format(str))
        return 1
    return output

def main():
    # * 1-3：トポロジーファイル：itpの読み込み
    # * ボンドの情報を読み込む．
    # *

    # * read input file

    # TODO :: hard code 
    fp=open("predict.inp",mode="r")

    inputs = []

    for line in fp.readlines():
        print(line.strip().split('='))
        inputs.append(line.strip().split('=')) # space/改行などを削除
    print("inputs :: {}".format(input))

    model_dir=find_input(inputs,"model_dir")
    # stdoutfile=find_input(inputs,"stdoutfile")
    desc_dir=find_input(inputs,"desc_dir")
    itpfilename=find_input(inputs,"itpfilename")

    
    import ml.atomtype
    itp_data=ml.atomtype.read_itp(itpfilename)
    bonds_list=itp_data.bonds_list
    NUM_MOL_ATOMS=itp_data.num_atoms_per_mol
    atomic_type=itp_data.atomic_type


    '''
    # * ボンドの情報設定
    # * 基本的にはitpの情報通りにCH，COなどのボンド情報を割り当てる．
    # * ボンドindexの何番がどのボンドになっているかを調べる．
    # * ベンゼン環だけは通常のC-C，C=Cと区別がつかないのでそこは手動にしないとダメかも．

    このボンド情報でボンドセンターの学習を行う．
    '''

    # ring_bonds = double_bonds_pairs
    ring_bonds = []

    # ボンド情報の読み込み(2022/12/20 作成)
    import ml.atomtype
    # importlib.reload(ml.atomtype)
    ch_bonds = itp_data.ch_bond
    co_bonds = itp_data.co_bond
    oh_bonds = itp_data.oh_bond
    cc_bonds = itp_data.cc_bond

    ring_bond_index = itp_data.ring_bond_index
    ch_bond_index   = itp_data.ch_bond_index
    co_bond_index   = itp_data.co_bond_index
    oh_bond_index   = itp_data.oh_bond_index
    cc_bond_index   = itp_data.cc_bond_index

    print(" ================== ")
    print(" ring_bond_index ", ring_bond_index)
    print(" ch_bond_index   ", ch_bond_index)
    print(" oh_bond_index   ", oh_bond_index)
    print(" co_bond_index   ", co_bond_index)
    print(" cc_bond_index   ", cc_bond_index)
    # <<<<<<<<<  bond info 読み込み ここまで <<<<<<<<<<


    import torch       # ライブラリ「PyTorch」のtorchパッケージをインポート
    import torch.nn as nn  # 「ニューラルネットワーク」モジュールの別名定義

    # TODO :: hardcode :: nfeatures :: ここはちょっと渡し方が難しいかも．
    nfeatures = 288
    print(" nfeatures :: ", nfeatures )
    
    # 定数（モデル定義時に必要となるもの）
    INPUT_FEATURES = nfeatures    # 入力（特徴）の数： 記述子の数
    LAYER1_NEURONS = 100     # ニューロンの数
    LAYER2_NEURONS = 100     # ニューロンの数
    #LAYER3_NEURONS = 200     # ニューロンの数
    #LAYER4_NEURONS = 100     # ニューロンの数
    OUTPUT_RESULTS = 3      # 出力結果の数： 3
    
    # torch.nn.Moduleによるモデルの定義
    class WFC(nn.Module):
        def __init__(self):
            super().__init__()
            
            # バッチ規格化層
            #self.bn1 = nn.BatchNorm1d(INPUT_FEATURES) #バッチ正規化
            
            # 隠れ層：1つ目のレイヤー（layer）
            self.layer1 = nn.Linear(
                INPUT_FEATURES,                # 入力ユニット数（＝入力層）
                LAYER1_NEURONS)                # 次のレイヤーの出力ユニット数
        
            # バッチ規格化層
            #self.bn2 = nn.BatchNorm1d(LAYER1_NEURONS) #バッチ正規化   
            
            # 隠れ層：2つ目のレイヤー（layer）
            self.layer2 = nn.Linear(
                LAYER1_NEURONS,                # 入力ユニット数（＝入力層）
                LAYER2_NEURONS)                # 次のレイヤーの出力ユニット数
            
            # バッチ規格化層
            #self.bn3 = nn.BatchNorm1d(LAYER2_NEURONS) #バッチ正規化   
            
            # 隠れ層：3つ目のレイヤー（layer）
            #self.layer3 = nn.Linear(
            #    LAYER2_NEURONS,                # 入力ユニット数（＝入力層）
            #    LAYER3_NEURONS)                # 次のレイヤーの出力ユニット数
            
            ## 隠れ層：4つ目のレイヤー（layer）
            #self.layer4 = nn.Linear(
            #    LAYER3_NEURONS,                # 入力ユニット数（＝入力層）
            #    LAYER4_NEURONS)                # 次のレイヤーの出力ユニット数
            
            # 出力層
            self.layer_out = nn.Linear(
                LAYER2_NEURONS,                # 入力ユニット数
                OUTPUT_RESULTS)                # 出力結果への出力ユニット数

        def forward(self, x):
                
            # フォワードパスを定義
            #x = self.bn1(x) #バッチ規格化
            x = nn.functional.leaky_relu(self.layer1(x))  
            #x = self.bn2(x) #バッチ規格化
            x = nn.functional.leaky_relu(self.layer2(x))  
            #x = self.bn3(x) #バッチ規格化
            #x = nn.functional.leaky_relu(self.layer3(x))  
            #x = nn.functional.leaky_relu(self.layer4(x))  
            x = self.layer_out(x)  # ※最終層は線形
            
            return x
    
    # モデル（NeuralNetworkクラス）のインスタンス化（これは絶対に必要）
    model_ring = WFC()
    model_ch = WFC()
    model_co = WFC()
    model_oh = WFC()
    model_o = WFC()
    
    from torchinfo import summary
    summary(model=model_ring)
    
    # 
    # * モデルをロードする場合はこれを利用する
    
    # model_dir="model_train40percent/"
    # model_ring.load_state_dict(torch.load('model_ring_weight.pth'))
    model_ch.load_state_dict(torch.load(model_dir+'model_ch_weight4.pth'))
    model_co.load_state_dict(torch.load(model_dir+'model_co_weight4.pth'))
    model_oh.load_state_dict(torch.load(model_dir+'model_oh_weight4.pth'))
    model_o.load_state_dict(torch.load(model_dir+'model_o_weight4.pth'))


    #
    # * 全データを再予測させる．
    
    #GPUが使用可能か確認
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    # 一旦モデルをcpuへ
    model_ch_2   = model_ch.to(device)
    model_oh_2   = model_oh.to(device)
    model_co_2   = model_co.to(device)
    model_o_2    = model_o.to(device)

    #
    # * ここから予測させる，すなわちここからデータをロードして並列化

    def predict_dipole(fr,desc_dir):
        #
        # * 機械学習用のデータを読み込む
        # *
        #
        import numpy as np
        
        # directory="merge_20230322/"
        
        # ring
        # descs_X_ring = np.loadtxt('Descs_ring.csv',delimiter=',')
        # data_y_ring = np.loadtxt('data_y_ring.csv',delimiter=',')
        
        # CHボンド
        descs_X_ch = np.loadtxt(desc_dir+'Descs_'+str(fr)+'_ch.csv',delimiter=',')
        # COボンド
        descs_X_co = np.loadtxt(desc_dir+'Descs_'+str(fr)+'_co.csv',delimiter=',')
        # OHボンド
        descs_X_oh = np.loadtxt(desc_dir+'Descs_'+str(fr)+'_oh.csv',delimiter=',')
        # Oローンペア
        descs_X_o =  np.loadtxt(desc_dir+'Descs_'+str(fr)+'_o.csv',delimiter=',')
        
        
        # 予測
        y_pred_ch  = model_ch_2(X_ch.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()
        y_pred_co  = model_co_2(X_co.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()
        y_pred_oh  = model_oh_2(X_oh.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()
        y_pred_o   = model_o_2(X_o.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()
    
        # 最後にreshape
        # TODO : hard code (酸素の数)
        num_o_bonds = 1
        y_pred_ch = y_pred_ch.reshape((NUM_MOL*len(ch_bond_index),3))
        y_pred_co = y_pred_co.reshape((NUM_MOL*len(co_bond_index),3))
        y_pred_oh = y_pred_oh.reshape((NUM_MOL*len(oh_bond_index),3))
        y_pred_o  = y_pred_o.reshape((NUM_MOL*num_o_bonds,3))
    
        #予測したモデルを使ったUnit Cellの双極子モーメントの計算
        sum_dipole=np.sum(y_pred_ch,axis=0)+np.sum(y_pred_oh,axis=0)+np.sum(y_pred_co,axis=0)+np.sum(y_pred_o,axis=0)
        return sum_dipole
        
    import joblib
            
    result_dipole = joblib.Parallel(n_jobs=-1, verbose=50)(joblib.delayed(predict_dipole)(fr) for fr in range(50001))
    result_dipole = np.array(result_dipole)
    np.save("result_dipole.npy",result_dipole)
    return result_dipole

if __name__ == '__main__':
    main()


import torch       # ライブラリ「PyTorch」のtorchパッケージをインポート
import torch.nn as nn  # 「ニューラルネットワーク」モジュールの別名定義


class WFC(nn.Module):
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
    
    
    
# torch.nn.Moduleによるモデルの定義
class NET(nn.Module):
    nfeatures = 288 # TODO :: hard code 4*12*6=288 # len(train_X_ch[0][0])
    print(" nfeatures :: ", nfeatures )
    
    M = 20 
    Mb= 6
    
    #Embedding Net 
    nfeatures_enet = int(nfeatures/4) # 72
    print(nfeatures_enet)
    # 定数（モデル定義時に必要となるもの）
    INPUT_FEATURES_enet = nfeatures_enet      # 入力（特徴）の数： 記述子の数
    LAYER1_NEURONS_enet = 50             # ニューロンの数
    LAYER2_NEURONS_enet = 50             # ニューロンの数
    OUTPUT_RESULTS_enet = M*nfeatures_enet    # 出力結果の数： 
    
    #Fitting Net 
    nfeatures_fnet = int(M*Mb) 
    print(nfeatures_fnet)
    # 定数（モデル定義時に必要となるもの）
    INPUT_FEATURES_fnet = nfeatures_fnet    # 入力（特徴）の数： 記述子の数
    LAYER1_NEURONS_fnet = 50     # ニューロンの数
    LAYER2_NEURONS_fnet = 50     # ニューロンの数
    OUTPUT_RESULTS_fnet = M      # 出力結果の数：
    
    def __init__(self):
        super().__init__()
        
        ##### Embedding Net #####
        # 隠れ層：1つ目のレイヤー（layer）
        self.Enet_layer1 = nn.Linear(
            INPUT_FEATURES_enet,                # 入力ユニット数（＝入力層）
            LAYER1_NEURONS_enet)                # 次のレイヤーの出力ユニット数
        
        # 隠れ層：2つ目のレイヤー（layer）
        self.Enet_layer2 = nn.Linear(
            LAYER1_NEURONS_enet,                # 入力ユニット数
            LAYER2_NEURONS_enet)                # 次のレイヤーの出力ユニット数
        
        # 出力層
        self.Enet_layer_out = nn.Linear(
            LAYER2_NEURONS_enet,                # 入力ユニット数
            OUTPUT_RESULTS_enet)                # 出力結果への出力ユニット数
        
        ##### Fitting net #####
        # 隠れ層：1つ目のレイヤー（layer）
        self.Fnet_layer1 = nn.Linear(
            INPUT_FEATURES_fnet,                # 入力ユニット数（＝入力層）
            LAYER1_NEURONS_fnet)                # 次のレイヤーの出力ユニット数
        
        # 隠れ層：2つ目のレイヤー（layer）
        self.Fnet_layer2 = nn.Linear(
            LAYER1_NEURONS_fnet,                # 入力ユニット数
            LAYER2_NEURONS_fnet)                # 次のレイヤーの出力ユニット数
        
        # 出力層
        self.Fnet_layer_out = nn.Linear(
            LAYER2_NEURONS_fnet,                # 入力ユニット数
            OUTPUT_RESULTS_fnet)                # 出力結果への出力ユニット数
        
    def forward(self, x):
                    
        #Si(1/Rをカットオフ関数で処理した値）のみを抽出する
        Q1 = x[:,::4]
        NB = Q1.size()[0]
        N  = Q1.size()[1]
        # Embedding Netに代入する 
        embedded_x = nn.functional.leaky_relu(self.Enet_layer1(Q1))  
        embedded_x = nn.functional.leaky_relu(self.Enet_layer2(embedded_x)) 
        embedded_x = self.Enet_layer_out(embedded_x)  # ※最終層は線形 
        #embedded_xを(ミニバッチデータ数)xMxN (N=MaxAt*原子種数)に変換
        embedded_x = torch.reshape(embedded_x,(NB,M,N ))
        #入力データをNB x N x 4 の行列に変形  
        matQ = torch.reshape(x,(NB,N,4))
        #Enetの出力との掛け算
        matT = torch.matmul(embedded_x, matQ)
        # matTの次元はNB x M x 4 となっている 
        #matSを作る(ハイパーパラメータMbで切り詰める)
        matS = matT[:,:Mb,:]
        #matSの転置行列を作る　→　NB x 4 x Mb となる 
        matSt = torch.transpose(matS, 1, 2)
        #matDを作る( matTとmatStの掛け算) →　NB x M x Mb となる 
        matD = torch.matmul(matT, matSt)
        #matDを１次元化する。matD全体をニューラルネットに入力したいので、ベクトル化する。 
        matD1 = torch.reshape(matD,(NB,M*Mb))
        # fitting Net に代入する 
        fitD = nn.functional.leaky_relu(self.Fnet_layer1(matD1))
        fitD = nn.functional.leaky_relu(self.Fnet_layer2(fitD)) 
        fitD = self.Fnet_layer_out(fitD)  # ※最終層は線形 
        # fitDの次元はNB x M となる。これをNB x 1 x Mの行列にする
        fitD3 = torch.reshape(fitD,(NB,1,M))
        # fttD3とmatTの掛け算 
        matW = torch.matmul(fitD3, matT) 
        # matWはNb x 1 x  4 になっている。これをNB x 4 の2次元にする
        matW2 = torch.reshape(matW,(NB,4))
        # はじめの要素はいらないので、切り詰めてx,y,z にする
        outW = matW2[:,1:]
        
        return outW
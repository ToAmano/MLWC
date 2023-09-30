def minibatch_train(test_rmse_list, train_rmse_list, test_loss_list, train_loss_list, model,dataloader_train, dataloader_valid, loss_function, epochs = 50, lr=0.0001, name="ch", modeldir="./"):
    '''
    * ミニバッチ学習の実施
    重要なこととして，どうもmodelをinputにして学習させると，どうも学習させた結果がそのまま残っているっぽい．
    つまり，return modelとしなくてもminibatch()を呼び出した後のmodel変数は
    
    return として
    test_rmses, train_loss
    を返す．
    
    重要ポイントとして，今のモデルではlossとRSMEが同じになる！！
    '''
    import torch
    import numpy as np
    #GPUが使用可能か確認
    # !! mac book 用にmpsを利用するように変更
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('mps')
    print("device (cpu or gpu ?) :: ",device)
    device = "cpu"     #もしCPUで学習させる場合はコメントアウトを外す
    model = model.to(device)
    
    # 最適化の設定
    torch.backends.cudnn.benchmark = True
    optimizer = torch.optim.Adam(model.parameters(),lr)     #最適化アルゴリズムの設定(adagradも試したがダメだった)
    
    # !! 学習率の動的変更
    # https://take-tech-engineer.com/pytorch-lr-scheduler/
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000,1000], gamma=0.1)
    
    # https://qiita.com/making111/items/21843f0aa41b486acc30
    torch.manual_seed(42)
    
    for epoch in range(epochs): 
        # epochごとにlossを格納するリスト
        loss_train = []
        loss_valid = []
        rsme_train = []
        rsme_valid = []
        
        model.train() # モデルを学習モードに変更
        for x, y in dataloader_train:                
            # !! FOR BN
            # BNの場合，xが[batch_size, nfeatures]の形になっているのでこのまま入れてみる．
            x = x.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()                   # 勾配情報を0に初期化
            y_pred = model(x)                       # 予測
            loss = loss_function(y_pred.reshape(y.shape), y)    # 損失を計算(shapeを揃える)
            np_loss = np.sqrt(np.mean((y_pred.to("cpu").detach().numpy()-y.to("cpu").detach().numpy())**2)) #損失のroot
            
            #print(loss)
            loss.backward()                         # 勾配の計算
            optimizer.step()                        # 勾配の更新
            optimizer.zero_grad()                   # https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
            scheduler.step()                        # !! 学習率の更新 
            rsme_train.append(np_loss)
            loss_train.append(loss.item())        
            del loss  # 誤差逆伝播を実行後、計算グラフを削除
        
        # テスト
        model.eval() # モデルを推論モードに変更 (BN)
        
        with torch.no_grad(): # https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
            for x,y in dataloader_valid:
                y_pred = model(x.to(device))                       # 予測
                loss = loss_function(y_pred.reshape(y.shape).to("cpu"), y)    # 損失を計算(shapeを揃える)
                np_loss = np.sqrt(np.mean((y_pred.to("cpu").detach().numpy()-y.detach().numpy())**2))  #損失のroot，RSMEと同じ
                rsme_valid.append(np_loss)
                loss_valid.append(loss.item())
        
        # バッチ全体でLoss値(のroot，すなわちRSME)を平均する
        ave_rsme_train = np.mean(np.array(rsme_train)) 
        ave_rsme_valid = np.mean(np.array(rsme_valid))
        ave_loss_train = np.mean(np.array(loss_train))
        ave_loss_valid = np.mean(np.array(loss_valid)) 
        
        test_rmse_list.append(ave_rsme_valid)
        train_rmse_list.append(ave_rsme_train)
        test_loss_list.append(ave_loss_valid)
        train_loss_list.append(ave_loss_train)
        print('epoch=', epoch+1, ' loss(ave_batch)=', ave_loss_train, ' loss(ave_test)=', ave_loss_valid, ' loss(ave_batch_np)=', ave_rsme_train, 'RMS Error(test)=', ave_rsme_valid)  
        
        # モデルの一時保存
        torch.save(model.state_dict(), modeldir+'model_'+name+'_weight_tmp.pth')
    return test_rmse_list, train_rmse_list, test_loss_list, train_loss_list


def calculate_final_dipoles(model, dataset):
    '''
    学習完了したモデルを利用して，train，test全データの双極子を計算する．
    メモリオーバーフローの対策として，別途データローダーを作成する（shuffleしないことで順番を保持する）．
    '''
    #ミニバッチ学習用のデータセットを構築する
    import torch
    import numpy as np
    ## DataSetクラスのインスタンスを作成
    # dataset_ch = DataSet(train_X_ch,true_y_ch)
    # dataset_ch_valid = DataSet(test_X_ch,test_y_ch)
    
    # datasetのサイズを取得
    datasize = len(dataset)
    # datasize*3の配列を確保?
    y_pred_list = [] #np.zeros(datasize*3).reshape(-1,3)
    y_true_list = []
    # datasetをDataLoaderの引数とすることでミニバッチを作成．
    # ! num_workers=0としないと動いてくれないので要注意！
    dataloader_infer = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False,drop_last=False, pin_memory=True, num_workers=0)
    
    #GPUが使用可能か確認
    # !! mac book 用にmpsを利用するように変更
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('mps')
    print("device (cpu or gpu ?) :: ",device)
    device = "cpu"     #もしCPUで学習させる場合はコメントアウトを外す
    model = model.to(device)
    # 推論
    model.eval() # モデルを推論モードに変更 (BN)
    with torch.no_grad(): # https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
        for x,y in dataloader_infer:
            y_pred = model(x.to(device)).to("cpu").detach().numpy()  # 予測
            for dipole_pred,dipole_true in zip(y_pred,y.detach().numpy()):
                y_pred_list.append(dipole_pred)
                y_true_list.append(dipole_true)
    # 整形
    y_pred_list = np.array(y_pred_list) #.reshape(-1,3)
    y_true_list = np.array(y_true_list) #.reshape(-1,3)

    
    return y_pred_list, y_true_list

def save_final_dipoles():
    '''
    学習結果のデータを保存する？
    （関数としての必要性が疑問）
    '''
    return 0

def save_model_cc(model, modeldir="./", name="cc"):
    '''
    C++用にモデルを保存する関数
    '''
    import torch
    # 学習時の入力サンプル
    device="cpu"
    example_input = torch.rand(1,model.nfeatures).to(device) # model.nfeatures=288

    # 学習済みモデルのトレース
    model_tmp = model.to(device) # model自体のdeviceを変えないように別変数に格納
    model_tmp.eval() # ちゃんと推論モードにする！！
    traced_net = torch.jit.trace(model_tmp, example_input)
    # 変換モデルの出力
    traced_net.save(modeldir+"model_"+name+".pt")
    return 0
    
    
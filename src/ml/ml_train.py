import torch
import torch.nn as nn  # 「ニューラルネットワーク」モジュールの別名定義
import logging
import os
import numpy as np
from typing import Callable, Optional, Union, Tuple, List
import ml.ml_dataset
from torch.utils.data.dataset import Subset
import time


class Trainer:
    def __init__(self, 
                model,
                device: str = "cuda" if torch.cuda.is_available() else "cpu",
                batch_size: int = 32,
                validation_batch_size: int = 32,
                max_epochs: int = 1000000,
                learning_rate: float = 1e-2,
                lr_scheduler_name: str = "none",
                lr_scheduler_kwargs: Optional[dict] = None,
                optimizer_name: str = "Adam",
                optimizer_kwargs: Optional[dict] = None, 
                n_train: Optional[int] = None,
                n_val: Optional[int] = None,
                modeldir:str = "./",
                restart: Optional[bool] = False):
        
        # import instance variables
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.validation_batch_size = validation_batch_size
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.lr_scheduler_name = lr_scheduler_name
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.optimier_name = optimizer_name
        self.optimier_kwargs = optimizer_kwargs
        self.n_train = n_train
        self.n_val   = n_val
        self.modeldir = modeldir # 保存するディレクトリ
        self.restart  = restart  # Trueの場合，以前の計算から再スタート
        
        # other instance variables 
        self.valid_rmse_list  = []
        self.train_rmse_list = []
        self.valid_loss_list  = []
        self.train_loss_list = []
        self.steps: int = 0  # total steps
        self.iepoch: int = 0 # total epochs
        self.best_epoch = 0
        
        # related to previous run
        self.previous_maxstep:int = -1
        
        # batch loss
        # TODO :: ここは完全にクラス化してもっと洗練された実装にしておきたい
        self.epoch_valid_loss = []
        self.epoch_train_loss = []

        
        # if not exist self.modeldir, mkdir it
        if not os.path.isdir(self.modeldir):
            os.makedirs(self.modeldir)
        self.logger.info(f"model data will be saved to {self.modeldir}")
        
        # generator
        self.dataset_rng = torch.Generator()
        
        # initialize training states
        self.best_metrics = float("inf")
        self.best_epoch = 0
        self.iepoch = 0 # -1 if self.report_init_validation else 0
        
        # print modelinfo
        from torchinfo import summary
        print(summary(model=self.model))
        
        # set loss function(損失関数)
        self.lossfunction = nn.MSELoss()  # 損失関数：平均二乗誤差
        
        # model initialize 
        self.init_model()
        
        # optimizer/scheduler
        self.init_optimizer_scheduler()
        
        # load previous run information
        if self.restart == True:
            self.get_previous_info()
            self.read_from_previous_run() # 既存ファイルがある場合，前回の結果を読み出し
            # self.iepoch = 
            
        

    @property
    def logger(self):
        # return logging.getLogger(self.logfile)
        return logging.getLogger("Trainer")


    @property
    def epoch_logger(self):
        return logging.getLogger(self.epoch_log)

    @property
    def init_epoch_logger(self):
        return logging.getLogger(self.init_epoch_log)
        
        
    def init_model(self):
        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # device = torch.device('mps') # mac book 用にmpsを利用するように変更
        self.logger.info(f"Torch device (cpu or cuda gpu or m1 mac gpu): {self.device}")
        self.model = self.model.to(self.device) # move to model

    def init_optimizer_scheduler(self):
        # 最適化の設定
        # TODO :: nequipでは，init_objects関数内で，instantiate_from_cls_nameという関数で実装している．
        # TODO :: この方法はかなりスマートなので導入したい．
        torch.backends.cudnn.benchmark = True
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate) #最適化アルゴリズムの設定(adagradも試したがダメだった)
    
        # !! 学習率の動的変更
        # TODO :: 変数を変更する
        # https://take-tech-engineer.com/pytorch-lr-scheduler/
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[1000,1000], gamma=0.1)

    def set_dataset(self,dataset:ml.ml_dataset.DataSet_custom,validation_dataset: Optional[ml.ml_dataset.DataSet_custom] = None):
        # total length of dataset        
        total_n = len(dataset)
        # 元のデータセットから，訓練データ数，validationデータ数に応じたデータを取り出す．
        if self.n_train is None or self.n_val is None:
            self.logger.warning(" n_train or n_val is not set.")
            self.logger.warning(" automatically 10% for validation and 90% for training ")
            self.n_train = int(0.9*total_n)
            self.n_val   = int(0.1*total_n)
            

        if validation_dataset is None: # val_datasetがない場合はdatasetから両方サンプルする
            if (self.n_train + self.n_val) > total_n:
                raise ValueError(
                    "too little data for training and validation. please reduce n_train and n_val"
                )

            idcs = torch.randperm(total_n, generator=self.dataset_rng)

            self.train_idcs = idcs[: self.n_train]
            self.val_idcs = idcs[self.n_train : self.n_train + self.n_val]
        else: # validation_datasetがあれば別々にsampleする
            if self.n_train > len(dataset):
                raise ValueError("Not enough data in dataset for requested n_train")
            if self.n_val > len(validation_dataset):
                raise ValueError(
                    "Not enough data in validation dataset for requested n_val"
                )

            self.train_idcs = torch.randperm(
                len(dataset), generator=self.dataset_rng
            )[: self.n_train]
            self.val_idcs = torch.randperm(
                len(validation_dataset), generator=self.dataset_rng
            )[: self.n_val]

        if validation_dataset is None:
            validation_dataset = dataset

        self.logger.info(f" n_traing ( number of training  data): {self.n_train}")
        self.logger.info(f" n_val    ( number of validatin data): {self.n_val}")

        # torch_geometric datasets inherantly support subsets using `index_select`
        # self.dataset_train = dataset.index_select(self.train_idcs)
        # self.dataset_val = validation_dataset.index_select(self.val_idcs)

        self.dataset_train = Subset(dataset,self.train_idcs)
        self.dataset_valid = Subset(validation_dataset,self.val_idcs)

        # dataset_train, dataset_valid = torch.utils.data.random_split(dataset=dataset, lengths=[len(dataset)-10000, 10000], generator=torch.Generator().manual_seed(42))

        # データローダーの定義
        self.dataloader_train = torch.utils.data.DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True,drop_last=True, pin_memory=True, num_workers=0)
        self.dataloader_valid = torch.utils.data.DataLoader(self.dataset_valid, batch_size=self.validation_batch_size, shuffle=True,drop_last=True, pin_memory=True, num_workers=0)

    def read_from_previous_run(self):
        # 既存ファイルがある場合，前回の結果を読み出し
        cptfile = f"{self.modeldir}/model_{self.model.modelname}_out_tmp{self.previous_maxstep}.cpt"
        if os.path.isfile(cptfile) == True:
            self.logger.info(" ---------------- ")
            self.logger.info(" cpt file exist :: load previous data !!")
            self.logger.info(" ---------------- ")
            cpt = torch.load(cptfile)
            stdict_m = cpt['model_state_dict']
            stdict_o = cpt['opt_state_dict']
            stdict_s = cpt['scheduler_state_dict']
            self.model.load_state_dict(stdict_m)
            self.optimizer.load_state_dict(stdict_o)
            self.scheduler.load_state_dict(stdict_s)
            
    def get_previous_info(self):
        # 既存ファイルから読み込む場合，最新のデータを調べる
        # ファイル のみ
        filenames = [int(f.name.removeprefix(f"model_{self.model.modelname}_out_tmp").removesuffix(".cpt")) for f in os.scandir(self.modeldir) if f.is_file() and f"model_{self.model.modelname}_out_tmp" in f.name]
        self.logger.info(filenames)
        # 数字の中で最も大きいものを取得
        self.previous_maxstep = np.max(np.array(filenames))
        self.logger.info(f"Previous run goes to {self.previous_maxstep} step")
        # !! update iepoch
        self.iepoch = self.previous_maxstep
        
        
        
    
    
    def train(self):
        # 実際のtrainingを行う場所
        # 個々の部品は別途定義してある
        
        # TODO ここも実装
        # if getattr(self, "dl_train", None) is None:
        #     raise RuntimeError("You must call `set_dataset()` before calling `train()`")
        # if not self._initialized:
        #     self.init()

        # self.init_log()
        # self.wall = perf_counter()
        # self.previous_cumulative_wall = self.cumulative_wall
        # self.init_metrics()
        # 繰り返し計算
        while not self.stop_condition:
            self.epoch_step() # epoch_stepがbatch_stepを含む形になっている
            self.end_of_epoch_save()
        # for callback in self._final_callbacks:
        #    callback(self)
        # self.final_log()
        # self.save()
        # finish_all_writes()
        
        # 全てのモデルを保存
        self.save_model_all()

    def epoch_step(self):
        '''
        1 epochのtrain/validationを行う．
        すなわち，dataloaderにあるデータをすべて使って推論する．
        '''
        
        # 時間計測        
        start_time = time.time()  # 現在時刻（処理開始前）を取得

        # 推論
        self.model.train() # モデルを学習モードに変更
        for data in self.dataloader_train: 
            self.logger.debug("start batch train")
            if data[0].dim() == 3: # 3次元の場合[NUM_BATCH,NUM_BOND,288]はデータを整形する
                # TODO :: torch.reshape(data[0], (-1, 288)) does not work !!
                for data_1 in zip(data[0],data[1]):
                    self.logger.debug(f" DEBUG :: data_1[0].shape = {data_1[0].shape} : data_1[1].shape = {data_1[1].shape}")
                    self.batch_step(data_1,validation=False)
                
            if data[0].dim() == 2: # 2次元の場合はそのまま
                # print("start batch train")
                self.batch_step(data,validation=False)
            
        # テスト
        self.model.eval() # モデルを推論モードに変更 (BN)
        with torch.no_grad(): # https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
            for data in self.dataloader_valid:
                self.logger.debug("start batch valid")
                if data[0].dim() == 3: # 3次元の場合[NUM_BATCH,NUM_BOND,288]はデータを整形する
                    # TODO :: torch.reshape(data[0], (-1, 288)) does not work !!
                    for data_1 in zip(data[0],data[1]):
                        self.logger.debug(f" DEBUG :: data_1[0].shape = {data_1[0].shape} : data_1[1].shape = {data_1[1].shape}")
                        self.batch_step(data_1,validation=True)
                if data[0].dim() == 2: # 2次元の場合はそのまま
                    self.batch_step(data,validation=True)
        
        # バッチ全体でLoss値(のroot，すなわちRSME)を平均する
        # TODO :: ここはもう少し良い実装を考えたい
        self.logger.debug(f" number of n_train/batch size ( iteration number of each step): {int(self.n_train/self.batch_size)} {int(self.n_val/self.validation_batch_size)}")
        ave_rmse_train = np.mean(np.array(self.train_rmse_list[-int(self.n_train/self.batch_size):])) 
        ave_rmse_valid = np.mean(np.array(self.valid_rmse_list[-int(self.n_val/self.validation_batch_size):]))
        ave_loss_train = np.mean(np.array(self.train_loss_list[-int(self.n_train/self.batch_size):]))
        ave_loss_valid = np.mean(np.array(self.valid_loss_list[-int(self.n_val/self.validation_batch_size):])) 
        # epoch平均loss情報の保持
        self.epoch_valid_loss.append(ave_loss_valid)
        self.epoch_train_loss.append(ave_loss_train)
        # timer        
        end_time = time.time()  # 現在時刻（処理完了後）を取得
        time_diff = end_time - start_time  # 処理完了後の時刻から処理開始前の時刻を減算する
        self.logger.info(f"epoch= {self.iepoch+1} : time= {time_diff} [s] : loss(train)= {ave_loss_train} : loss(valid)= {ave_loss_valid} : RMSE[D](train)= {ave_rmse_train} : RMSE[D](valid)= {ave_rmse_valid}")
        
        # update epoch step
        self.iepoch += 1
        
        # self.end_of_epoch_log()


    def end_of_epoch_save(self):
        """
        save model and trainer details at each epoch ( for restarting)
        """
        # モデルの一時保存
        if self.previous_maxstep < 0: # 前回から読み込まない場合
            torch.save(self.model.state_dict(), f"{self.modeldir}/model_{self.model.modelname}_weight_tmp_{str(self.iepoch)}.pth")
        else: # 前回から読み込む場合
            torch.save(self.model.state_dict(), f"{self.modeldir}/model_{self.model.modelname}_weight_tmp_{str(self.iepoch+self.previous_maxstep)}.pth")
            
        # 学習状態の一時保存
        torch.save({'iter':          self.iepoch,
            'model_state_dict':      self.model.state_dict(),
            'opt_state_dict':        self.optimizer.state_dict(),
            'scheduler_state_dict' : self.scheduler.state_dict(),
            'loss'                 : self.train_loss_list,
            }, self.modeldir+'/model_'+self.model.modelname+'_out_tmp'+str(self.iepoch)+'.cpt')
        # print("model is saved !! ", self.modeldir+'/model_'+self.model.modelname+'_out_tmp'+str(self.iepoch)+'.cpt')
        
        # C++ version model save
        save_model_cc(self.model, modeldir=self.modeldir, name=self.model.modelname+'_tmp'+str(self.iepoch))
        # >>> end


    def batch_step(self, data, validation:bool=False):
        '''
        data:: これが実際に計算するデータで，dataloaderから引っ張ってきたもの
        '''
        if validation:
            self.model.eval()
        else:
            self.model.train()

        # datasetは[x,y]の形で返すようになっている
        x = data[0].to(self.device)
        y = data[1].to(self.device)
        
        if not validation:
            self.optimizer.zero_grad()                   # 勾配情報を0に初期化
            y_pred = self.model(x)                       # 予測
            loss = self.lossfunction(y_pred.reshape(y.shape), y)    # 損失を計算(shapeを揃える)
            # np_loss = np.sqrt(np.mean((y_pred.to("cpu").detach().numpy()-y.to("cpu").detach().numpy())**2)) #損失のroot
            
            #print(loss)
            loss.backward()                         # 勾配の計算
            self.optimizer.step()                        # 勾配の更新
            self.optimizer.zero_grad()                   # https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
            self.scheduler.step()                        # !! 学習率の更新 
            # self.rsme_train.append(np_loss)
            # self.loss_train.append(loss.item())  
            self.train_rmse_list.append(np.sqrt(loss.item()))
            self.train_loss_list.append(loss.item())      
            del loss  # 誤差逆伝播を実行後、計算グラフを削除

        else: # validation
            with torch.no_grad(): # https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
                y_pred = self.model(x.to(self.device))                       # 予測
                loss = self.lossfunction(y_pred.reshape(y.shape).to("cpu"), y.to("cpu"))    # 損失を計算(shapeを揃える)
                # np_loss = np.sqrt(np.mean((y_pred.to("cpu").detach().numpy()-y.detach().numpy())**2))  #損失のroot，RSMEと同じ
                # self.rsme_valid.append(np_loss)
                # self.loss_valid.append(loss.item())
                self.valid_rmse_list.append(np.sqrt(loss.item()))
                self.valid_loss_list.append(loss.item())

    def save_model_all(self):
        '''
        モデルを全て保存する．
        '''
        import torch
        # モデルの重み保存
        print(" model is saved to {} at {}".format('model_'+self.model.modelname+'_weight.pth',self.modeldir))
        torch.save(self.model.state_dict(), self.modeldir+'/model_'+self.model.modelname+'_weight.pth') # fin
        # モデル全体保存
        # https://take-tech-engineer.com/pytorch-model-save-load/#toc3
        print(" model is saved to {} at {}".format('model_'+self.model.modelname+'_all.pth',self.modeldir))
        torch.save(self.model, self.modeldir+'/model_'+self.model.modelname+'_all.pth')    
        ## c++用のtorch scriptを保存
        self.save_model_cc()
        return 0
    
    def save_model_cc(self):
        '''
        C++用にモデルを保存する関数
        '''
        import torch
        # 学習時の入力サンプル
        device="cpu"
        example_input = torch.rand(1,self.model.nfeatures).to(device) # model.nfeatures=288

        # 学習済みモデルのトレース
        model_tmp = self.model.to(device) # model自体のdeviceを変えないように別変数に格納
        model_tmp.eval() # ちゃんと推論モードにする！！
        traced_net = torch.jit.trace(model_tmp, example_input)
        # 変換モデルの出力
        print(" model is saved to {} at {}".format('model_'+self.model.modelname+'.pt',self.modeldir))
        traced_net.save(self.modeldir+"/model_"+self.model.modelname+".pt")
        return 0
        
    def save_prediction_result():
        print("TEST")
        
    @property
    def stop_condition(self):
        """
        学習を止めるかどうかの判定に用いる．
        現状はepoch数がmaxに達したら停止する．
        """
        if self.iepoch >= self.max_epochs:
            self.stop_arg = "max epochs"
            return True
        return False





# ==========================
# 以下従来の関数型の実装
# ==========================

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
    import os
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
    
    # 既存ファイルがある場合，前回の結果を読み出し
    cptfile = modeldir+'model_'+name+'_out_tmp.cpt'
    if os.path.isfile(cptfile) == True:
        print(" ---------------- ")
        print(" cpt file exist :: load previous data !!")
        print(" ---------------- ")
        cpt = torch.load(cptfile)
        stdict_m = cpt['model_state_dict']
        stdict_o = cpt['opt_state_dict']
        stdict_s = cpt['scheduler_state_dict']
        model.load_state_dict(stdict_m)
        optimizer.load_state_dict(stdict_o)
        scheduler.load_state_dict(stdict_s)
    
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
        # 学習状態の一時保存
        torch.save({'iter': epoch,
            'model_state_dict': model.state_dict(),
            'opt_state_dict': optimizer.state_dict(),
            'scheduler_state_dict' : scheduler.state_dict(),
            'loss': train_loss_list,
            }, modeldir+'model_'+name+'_out_tmp.cpt')
        
    return test_rmse_list, train_rmse_list, test_loss_list, train_loss_list






def minibatch_train_with_scheduler(test_rmse_list, train_rmse_list, test_loss_list, train_loss_list, model, dataloader_train, dataloader_valid, loss_function, scheduler_dict, epochs = 50, lr=0.0001, name="ch", modeldir="./"):
    '''
    * ミニバッチ学習の実施
    重要なこととして，どうもmodelをinputにして学習させると，どうも学習させた結果がそのまま残っているっぽい．
    つまり，return modelとしなくてもminibatch()を呼び出した後のmodel変数は
    
    return として
    test_rmses, train_loss
    を返す．
    
    重要ポイントとして，今のモデルではlossとRSMEが同じになる！！
    
    1: 学習中のloss, RMSEを保存する．保存する量として
        1: batchごとのloss
        2: epochごとのloss（batch lossをbatchで平均化したもの）
    
    scheduler_dict :: start_factor=1, end_factor=0.1, total_iters=50
    '''
    import torch
    import numpy as np
    import os
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
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000,1000], gamma=0.1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(scheduler_dict) # start_factor=1, end_factor=0.1, total_iters=50)
    
    # 既存ファイルがある場合読み出し
    cptfile = modeldir+'model_'+name+'_out_tmp.cpt'
    if os.path.isfile(cptfile) == True:
        cpt = torch.load(cptfile)
        stdict_m = cpt['model_state_dict']
        stdict_o = cpt['opt_state_dict']
        stdict_s = cpt['scheduler_state_dict']
        model.load_state_dict(stdict_m)
        optimizer.load_state_dict(stdict_o)
        scheduler.load_state_dict(stdict_s)
    
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
        
        # 学習状態の一時保存
        torch.save({'iter': epoch,
            'model_state_dict': model.state_dict(),
            'opt_state_dict': optimizer.state_dict(),
            'scheduler_state_dict' : scheduler.state_dict(),
            'loss': train_loss_list,
            }, modeldir+'model_'+name+'_out_tmp.cpt')
                
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
    print(" model is saved to {} at {}".format('model_'+name+'.pt',modeldir))
    traced_net.save(modeldir+"/model_"+name+".pt")
    return 0
    

def save_model_all(model, modeldir:str, name:str="ch"):
    '''
    モデルを全て保存する．
    '''
    import torch
    # モデルの重み保存
    print(" model is saved to {} at {}".format('model_'+name+'_weight.pth',modeldir))
    torch.save(model.state_dict(), modeldir+'/model_'+name+'_weight.pth') # fin
    # モデル全体保存
    # https://take-tech-engineer.com/pytorch-model-save-load/#toc3
    print(" model is saved to {} at {}".format('model_'+name+'_all.pth',modeldir))
    torch.save(model, modeldir+'model_'+name+'_all.pth')    
    ## c++用のtorch scriptを保存
    save_model_cc(model, modeldir, name=name)
    return 0


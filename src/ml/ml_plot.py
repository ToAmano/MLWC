import torch
import numpy as np
import ml.ml_train

def plot_loss_class(test,train):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8,5),tight_layout=True) # figure, axesオブジェクトを作成
    scatter1=ax.scatter(np.arange(len(test)), test, label="test",alpha=0.2,color="#1f77b4", s=5)  # 描画
    scatter2=ax.scatter(np.arange(len(train)), train, label="train",alpha=0.2, color='#ff7f0e',s=5)

    # 各要素で設定したい文字列の取得
    xticklabels = ax.get_xticklabels()
    yticklabels = ax.get_yticklabels()
    xlabel="Step "
    ylabel="Loss [%]"
    
    # 各要素の設定を行うsetコマンド
    ax.set_xlabel(xlabel,fontsize=22)
    ax.set_ylabel(ylabel,fontsize=22)
    
    # ax.set_xlim(0,3)
    # ax.set_ylim(0,3)
    ax.grid()
    
    ax.tick_params(axis='x', labelsize=20 )
    ax.tick_params(axis='y', labelsize=20 )
    
    ax.tick_params(axis='x', labelsize=20 )
    ax.tick_params(axis='y', labelsize=20 )
    
    # ax.legend = ax.legend(*scatter.legend_elements(prop="colors"),loc="upper left", title="Ranking")
    
    lgnd=ax.legend(loc="upper left",fontsize=20)
    lgnd.legendHandles[0]._sizes = [30]
    lgnd.legendHandles[1]._sizes = [30]
    lgnd.legendHandles[0]._alpha = [1.0]
    lgnd.legendHandles[1]._alpha = [1.0]
    
    
    #pyplot.savefig("eps_real2.pdf",transparent=True)
    # plt.show()
    # fig.savefig(load_dir+"leaning_result.png")
    # ax.show()
    # fig.delaxes(ax)

    # plt.legend()
    # plt.show()
    return 0

def plot_loss(test_loss_list,train_loss_list,scale="normal")->int:
    '''
    学習ロスをプロットする関数
    test :: testデータのloss値/RSME値
    train :: trainデータのloss値/RSME値
    '''
    import matplotlib.pyplot as plt
    import numpy as np
    test_x=np.arange(len(test_loss_list))
    plt.plot(test_x,test_loss_list,label="test")
    train_x=np.arange(len(train_loss_list))
    plt.plot(train_x,train_loss_list,label="train")
    if scale == "log":
        plt.yscale("log")
    plt.legend()
    plt.title("loss of train/test")
    plt.xlabel("step")
    plt.ylabel("loss(%)")
    return 0

def plot_loss_log(test_loss_list,train_loss_list):
    '''
    学習ロスをプロットする関数
    '''
    plot_loss(test_loss_list,train_loss_list,scale="log")
    return 0


def plot_residure_train_valid(model, dataset_train, dataset_valid,limit:bool=True):
    '''
    学習結果をplotする関数．
    '''
    import matplotlib.pyplot as plt
    import numpy as np
    from ml.ml_train import calculate_final_dipoles
    
    # [size,3]の形でdipoleを返す
    train_pred_list, train_true_list = calculate_final_dipoles(model, dataset_train)
    valid_pred_list, valid_true_list = calculate_final_dipoles(model, dataset_valid)
    
    # RSMEを計算する
    rmse_train = np.sqrt(np.mean((train_true_list-train_pred_list)**2))
    rmse_valid = np.sqrt(np.mean((valid_true_list-valid_pred_list)**2))
    print(" RSME_train = {0}".format(rmse_train))
    print(" RSME_valid = {0}".format(rmse_valid))
    
    # matplotlibで複数のプロットをまとめる．
    # https://python-academia.com/matplotlib-multiplegraphs/
    # グラフを表示する領域を，figオブジェクトとして作成。
    fig = plt.figure(figsize = (15,5), facecolor='lightblue')
    
    #グラフを描画するsubplot領域を作成。
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)
    
    #各subplot領域にデータを渡す
    # TODO :: RSME，決定係数Rを同時に表示するようにする


    ax1.scatter(train_pred_list[:,0], train_true_list[:,0], alpha=0.03, label="train")    
    ax1.scatter(valid_pred_list[:,0], valid_true_list[:,0], alpha=0.03, label="valid")    
    
    ax2.scatter(train_pred_list[:,1], train_true_list[:,1], alpha=0.03, label="train")    
    ax2.scatter(valid_pred_list[:,1], valid_true_list[:,1], alpha=0.03, label="valid")    
    
    ax3.scatter(train_pred_list[:,2], train_true_list[:,2], alpha=0.03, label="train")    
    ax3.scatter(valid_pred_list[:,2], valid_true_list[:,2], alpha=0.03, label="valid")    

    #タイトル
    ax1.set_title("Dipole_x")
    ax2.set_title("Dipole_y")
    ax3.set_title("Dipole_z")

    #各subplotにxラベルを追加
    ax1.set_xlabel("ML dipole [D]")
    ax2.set_xlabel("ML dipole [D]")
    ax3.set_xlabel("ML dipole [D]")

    ax1.set_ylabel("CPMD dipole [D]")
    ax2.set_ylabel("CPMD dipole [D]")
    ax3.set_ylabel("CPMD dipole [D]")

    # 凡例表示
    # https://qiita.com/hnii2006/items/2db5312fe4a4365734d0
    legend1=ax1.legend(loc = 'upper left') #.get_frame().set_alpha(1.0)
    legend2=ax2.legend(loc = 'upper left') # .get_frame().set_alpha(1.0) 
    legend3=ax3.legend(loc = 'upper left') # .get_frame().set_alpha(1.0) 
    
    # grid表示
    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)

    plt.show()

    # if limit:
    #     plt.xlim(-2,2)
    #     plt.ylim(-2,2)
    #    rmse_train = np.sqrt(np.mean((x0[:,i]-y0[:,i])**2))
    #    rmse_test  = np.sqrt(np.mean((x1[:,i]-y1[:,i])**2))
    #    print("rmse(train):  {0}  / rmse(test):  {1}".format(rmse_train,rmse_test))

    return 0

def calculate_gaussian_kde(data_x,data_y):
    # https://runtascience.hatenablog.com/entry/2021/05/06/%E3%80%90Matplotlib%E3%80%91python%E3%81%A7%E5%AF%86%E5%BA%A6%E3%83%97%E3%83%AD%E3%83%83%E3%83%88%28Density_plot%29
    from scipy.stats import gaussian_kde
    # KDE probability
    x = data_x
    y = data_y
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    # zの値で並び替え→x,yも並び替える
    idx = z.argsort() 
    x, y, z = x[idx], y[idx], z[idx]
    return x,y,z

def plot_residure_density(model, dataset,limit:bool=True):
    '''
    学習結果をplotする関数．
    こちらではtrain/validの区別なく，全てのデータをまとめて，代わりにdensity mapで表示する
    '''
    import matplotlib.pyplot as plt
    import numpy as np
    from ml.ml_train import calculate_final_dipoles
    
    # [size,3]の形でdipoleを返す
    pred_list, true_list = calculate_final_dipoles(model, dataset)
    
    # RSMEを計算する
    rmse = np.sqrt(np.mean((true_list-pred_list)**2))
    print(" RSME_train = {0}".format(rmse))
    
    # matplotlibで複数のプロットをまとめる．
    # https://python-academia.com/matplotlib-multiplegraphs/
    # グラフを表示する領域を，figオブジェクトとして作成。
    fig = plt.figure(figsize = (15,5), facecolor='lightblue')
    
    #グラフを描画するsubplot領域を作成。
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)
    
    #各subplot領域にデータを渡す
    # TODO :: RSME，決定係数Rを同時に表示するようにする

    # KDE probability
    x,y,z = calculate_gaussian_kde(pred_list[:,0], true_list[:,0])
    im = ax1.scatter(x, y, c=z, s=50, cmap="jet")
    fig.colorbar(im)

    x,y,z = calculate_gaussian_kde(pred_list[:,1], true_list[:,1])
    im = ax2.scatter(x, y, c=z, s=50, cmap="jet")
    fig.colorbar(im)

    x,y,z = calculate_gaussian_kde(pred_list[:,2], true_list[:,2])
    im = ax3.scatter(x, y, c=z, s=50, cmap="jet")
    fig.colorbar(im)

    #各subplotにxラベルを追加
    ax1.set_xlabel("ML dipole [D]")
    ax2.set_xlabel("ML dipole [D]")
    ax3.set_xlabel("ML dipole [D]")

    ax1.set_ylabel("CPMD dipole [D]")
    ax2.set_ylabel("CPMD dipole [D]")
    ax3.set_ylabel("CPMD dipole [D]")

    # 凡例表示
    ax1.legend(loc = 'upper left') 
    ax2.legend(loc = 'upper left') 
    ax3.legend(loc = 'upper left') 

    # grid表示
    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)

    plt.show()

    # if limit:
    #     plt.xlim(-2,2)
    #     plt.ylim(-2,2)
    #    rmse_train = np.sqrt(np.mean((x0[:,i]-y0[:,i])**2))
    #    rmse_test  = np.sqrt(np.mean((x1[:,i]-y1[:,i])**2))
    #    print("rmse(train):  {0}  / rmse(test):  {1}".format(rmse_train,rmse_test))

    return 0


def plot_residure(y_pred_train,y_pred_test,true_y,test_y,limit:bool=True):
    '''
    学習結果をplotする関数．
    '''
    import matplotlib.pyplot as plt
    import numpy as np

    # <<< nfeaturesを使えないので却下
    #GPUが使用可能か確認
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # print(device)
    # y_pred_train= model(train_X.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()
    # y_pred_test= model(test_X.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()

    x0 = y_pred_train
    y0 = true_y.reshape(-1,3).detach().numpy()

    x1 = y_pred_test
    y1 = test_y.reshape(-1,3).detach().numpy() 
    
    # debug
    print("x0.shape :: ", x0.shape)
    print("y0.shape :: ", y0.shape)
    print("x1.shape :: ", x1.shape)
    print("y1.shape :: ", y1.shape)

    rmse0 = np.sqrt(np.mean((x0-y0)**2))
    rmse1 = np.sqrt(np.mean((x1-y1)**2))
    print(rmse0,rmse1)

    vector = ["x","y","z"]  

    for i,c in enumerate(vector) :
        plt.scatter(x0[:,i],y0[:,i],alpha=0.1,label="train")
        plt.scatter(x1[:,i],y1[:,i],alpha=0.03,label="test")
        if limit:
            plt.xlim(-2,2)
            plt.ylim(-2,2)
        rmse_train = np.sqrt(np.mean((x0[:,i]-y0[:,i])**2))
        rmse_test  = np.sqrt(np.mean((x1[:,i]-y1[:,i])**2))
        print("rmse(train):  {0}  / rmse(test):  {1}".format(rmse_train,rmse_test))
        #plt.title("This is a title")
        plt.xlabel("ANN predicted mu ")
        plt.ylabel("QE simulated mu ")
        plt.grid(True)
        plt.title(str(c))
        plt.legend()
        plt.show()
    return 0

def plot_norm(y_pred_train,y_pred_test,true_y,test_y,limit:bool=True,save="", title:str="ML_result"):
    '''
    学習結果をplotする関数．

    limit :: plotする区間の制限をかけるかかけないか．
    '''
    import matplotlib.pyplot as plt
    import numpy as np

    # <<< nfeaturesを使えないので却下
    #GPUが使用可能か確認
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # print(device)

    # y_pred_train= model(train_X.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()
    # y_pred_test= model(test_X.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()

    x0 = y_pred_train
    y0 = true_y.reshape(-1,3).detach().numpy()

    x1 = y_pred_test
    y1 = test_y.reshape(-1,3).detach().numpy() 
    
    # debug
    print("x0.shape :: ", x0.shape)
    print("y0.shape :: ", y0.shape)
    print("x1.shape :: ", x1.shape)
    print("y1.shape :: ", y1.shape)

    rmse0 = np.sqrt(np.mean((x0-y0)**2))
    rmse1 = np.sqrt(np.mean((x1-y1)**2))
    print(rmse0,rmse1)

    plt.scatter(np.linalg.norm(x0,axis=1),np.linalg.norm(y0,axis=1),alpha=0.2,s=5,label="train RMSE={}".format(rmse0))
    plt.scatter(np.linalg.norm(x1,axis=1),np.linalg.norm(y1,axis=1),alpha=0.2,s=5,label="test RMSE={}".format(rmse1))
    if limit:
        plt.xlim(0,4)
        plt.ylim(0,4)
    #plt.title("This is a title")
    plt.xlabel("ANN predicted mu ")
    plt.ylabel("QE simulated mu ")
    plt.grid(True)
    plt.title(title) # titleを指定できるように
    lgnd=plt.legend()
    lgnd.legendHandles[0]._sizes = [30]
    lgnd.legendHandles[1]._sizes = [30]
    lgnd.legendHandles[0]._alpha = [1.0]
    lgnd.legendHandles[1]._alpha = [1.0]
    plt.show()
    
    # if save ?
    if save != "":
        import os
        os.mkdir(save)
        np.savetxt(save+"/test_pred.txt", np.linalg.norm(x1,axis=1))
        np.savetxt(save+"/test_true.txt", np.linalg.norm(y1,axis=1))
        np.savetxt(save+"/train_pred.txt", np.linalg.norm(x0,axis=1))
        np.savetxt(save+"/train_true.txt", np.linalg.norm(y0,axis=1))
    return 0


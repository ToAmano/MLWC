import torch
import numpy as np

def plot_loss(test,train):
    '''
    学習ロスをプロットする関数
    '''
    import matplotlib.pyplot as plt
    import numpy as np
    test_x=np.arange(len(test))
    plt.plot(test_x,test,label="test")
    train_x=np.arange(len(train))
    plt.plot(train_x,train,label="train")
    plt.legend()
    plt.title("loss of train/test")
    plt.xlabel("step")
    plt.ylabel("loss(%)")
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

    plt.scatter(np.linalg.norm(x0,axis=1),np.linalg.norm(y0,axis=1),alpha=0.2,s=5,label="train")
    plt.scatter(np.linalg.norm(x1,axis=1),np.linalg.norm(y1,axis=1),alpha=0.2,s=5,label="test")
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


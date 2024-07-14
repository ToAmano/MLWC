



# =========================================================

def plot_vector(train_X,test_X,true_y,test_y,model):
    '''
    双極子の学習結果（ベクトル）をプロットする．
    '''
    import matplotlib.pyplot as plt
    
    #GPUが使用可能か確認
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    y_pred_train= model(train_X.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()
    y_pred_test= model(test_X.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()

    x0 = y_pred_train
    y0 = true_y.reshape(-1,3).detach().numpy()

    x1 = y_pred_test
    y1 = test_y.reshape(-1,3).detach().numpy() 

    rmse0 = np.sqrt(np.mean((x0-y0)**2))
    rmse1 = np.sqrt(np.mean((x1-y1)**2))
    print(rmse0,rmse1)

    vector = ["x","y","z"]  

    for i,c in enumerate(vector) :
        plt.scatter(x0[:,i],y0[:,i],alpha=0.1)
        plt.scatter(x1[:,i],y1[:,i],alpha=0.03)
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
        plt.show()
    return 0


def plot_norm(train_X,test_X,true_y,test_y,model):
    '''
    双極子の学習結果をベクトルのノルムを取ってプロットする．
    '''
    import matplotlib.pyplot as plt
    
    #GPUが使用可能か確認
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    y_pred_train= model(train_X.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()
    y_pred_test= model(test_X.reshape(-1,nfeatures).to(device)).to("cpu").detach().numpy()

    x0 = y_pred_train
    y0 = true_y.reshape(-1,3).detach().numpy()

    x1 = y_pred_test
    y1 = test_y.reshape(-1,3).detach().numpy() 

    rmse0 = np.sqrt(np.mean((x0-y0)**2))
    rmse1 = np.sqrt(np.mean((x1-y1)**2))
    print(rmse0,rmse1)

    # ノルムを求める．
    x0_norm = np.linalg.norm(x0,axis=1)
    y0_norm = np.linalg.norm(y0,axis=1)
    x1_norm = np.linalg.norm(x1,axis=1)
    y1_norm = np.linalg.norm(y1,axis=1)

    
    plt.scatter(x0_norm,y0_norm,alpha=0.1)
    plt.scatter(x1_norm,y1_norm,alpha=0.03)
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    rmse_train = np.sqrt(np.mean((x0_norm-y0_norm)**2))
    rmse_test  = np.sqrt(np.mean((x1_norm-y1_norm)**2))
    print("rmse(train):  {0}  / rmse(test):  {1}".format(rmse_train,rmse_test))
    #plt.title("This is a title")
    plt.xlabel("ANN predicted mu ")
    plt.ylabel("QE simulated mu ")
    plt.grid(True)
    plt.title(" Norm of Dipole Moment [D] ")
    plt.show()
    return 0



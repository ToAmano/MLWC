def divide_data(descs_X,data_y,a_test,a_train):
    '''
    記述子を訓練用とテスト用に分ける．
    input
    ------------
    a_test::テスト用配列
    a_train::訓練用配列
    '''
    
    import copy 
    #データ分割（訓練用とテスト用に分ける）
    test_X = descs_X[a_test]
    test_y = data_y[a_test]
    train_X = descs_X[a_train]
    true_y  = data_y[a_train]

    print("train_X,y:{0} {1} / test_X,y:{2} {3} ".format(len(train_X),len(true_y),len(test_X),len(test_y)))
    return test_X, test_y, train_X, true_y


#encoding=utf8
from sklearn.svm import LinearSVC
import pandas as pd

def linearsvc_predict(train_data,train_label,test_data):
    '''
    input:train_data(ndarray):训练数据
          train_label(ndarray):训练标签
    output:predict(ndarray):测试集预测标签
    '''
    #********* Begin *********#
    train_data=pd.read_csv('./step1/train_data.csv')
    train_label=pd.read_csv('./step1/train_label.csv')
    train_label=train_label['target']
    test_data=pd.read_csv('./step1/test_data.csv')
    svc=LinearSVC(C=1,max_iter=2000)
    svc.fit(train_data,train_label)
    svc.fit(train_data,train_label)


    #********* End *********#
    return svc.predict(test_data)
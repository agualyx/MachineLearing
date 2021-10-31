import numpy as np
import random
#构建感知机算法
class Perceptron(object):
    def __init__(self, learning_rate = 0.01, max_iter = 200):
        self.lr = learning_rate
        self.max_iter = max_iter
    def fit(self, data, label):
        '''
        input:data(ndarray):训练数据特征
              label(ndarray):训练数据标签
        output:w(ndarray):训练好的权重
               b(ndarry):训练好的偏置
        '''
        #编写感知机训练方法，w为权重，b为偏置
        self.w = np.array([1.]*data.shape[1])
        self.b = np.array([1.])
        #********* Begin *********#
        for i in range(self.max_iter):
            #用于标纸每一次迭代是否有纠错行为，若无纠错行为则正确
            flag=False
            for j in range(data.shape[0]):
                x=data[j]
                y=label[j]
                #如果数据预测错误，对w和b按照题目所给的公式进行修正
                if y*(x.dot(self.w)+self.b)<0:
                    self.w+=self.lr*x*y
                    self.b+=self.lr*y
                    flag=True
            #没有纠错，即数据全部正确时跳出循环
            if not flag:
                break
        #********* End *********#
    def predict(self, data):
        '''
        input:data(ndarray):测试数据特征
        output:predict(ndarray):预测标签
        '''
        #********* Begin *********#
        predict=[]
        for i in range(data.shape[0]):
            #根据损失函数预测
            if data[i]*self.w+self.b<0:
                predict.append(-1)
            else:
                predict.append(1)

        #********* End *********#
        return np.array(predict)
import numpy as np


class NaiveBayesClassifier(object):
    def __init__(self):
        '''
        self.label_prob表示每种类别在数据中出现的概率
        例如，{0:0.333, 1:0.667}表示数据中类别0出现的概率为0.333，类别1的概率为0.667
        '''
        self.label_prob = {}
        '''
        self.condition_prob表示每种类别确定的条件下各个特征出现的概率
        例如训练数据集中的特征为 [[2, 1, 1],
                              [1, 2, 2],a
                              [2, 2, 2],
                              [2, 1, 2],
                              [1, 2, 3]]
        标签为[1, 0, 1, 0, 1]
        那么当标签为0时第0列的值为1的概率为0.5，值为2的概率为0.5;
        当标签为0时第1列的值为1的概率为0.5，值为2的概率为0.5;
        当标签为0时第2列的值为1的概率为0，值为2的概率为1，值为3的概率为0;
        当标签为1时第0列的值为1的概率为0.333，值为2的概率为0.666;
        当标签为1时第1列的值为1的概率为0.333，值为2的概率为0.666;
        当标签为1时第2列的值为1的概率为0.333，值为2的概率为0.333,值为3的概率为0.333;
        因此self.label_prob的值如下：     
        {
            0:{
                0:{
                    1:0.5
                    2:0.5
                }
                1:{
                    1:0.5
                    2:0.5
                }
                2:{
                    1:0
                    2:1
                    3:0
                }
            }
            1:
            {
                0:{
                    1:0.333
                    2:0.666
                }
                1:{
                    1:0.333
                    2:0.666
                }
                2:{
                    1:0.333
                    2:0.333
                    3:0.333
                }
            }
        }
        '''
        self.condition_prob = {}

    def fit(self, feature, label):
        '''
        对模型进行训练，需要将各种概率分别保存在self.label_prob和self.condition_prob中
        :param feature: 训练数据集所有特征组成的ndarray
        :param label:训练数据集中所有标签组成的ndarray
        :return: 无返回
        '''

        # ********* Begin *********#
        #提取出为坏瓜的索引
        feature0_index=np.where(label==0)
        #提取出为好瓜的索引
        feature1_index=np.where(label==1)
        #获得坏瓜的数据集集合
        feature0=feature[feature0_index]
        #获得好瓜的数据集集合
        feature1=feature[feature1_index]
        #统计坏瓜各种性状出现的概率
        self.label_prob[0]=len(feature0)/len(feature)
        self.label_prob[1]=len(feature1)/len(feature)
        sum0={}
        for f0 in feature0:
            for i in range(len(f0)):
                if i in sum0:
                    sum0[i][f0[i]] = sum0[i].get(f0[i], 0) + 1
                else:
                    sum0[i] = {}
                    sum0[i][f0[i]] = sum0[i].get(f0[i], 0) + 1
        sum1={}
        for f1 in feature1:
            for i in range(len(f1)):
                if i in sum1:
                    sum1[i][f1[i]] = sum1[i].get(f1[i], 0) + 1 / len(feature1)
                else:
                    sum1[i] = {}
                    sum1[i][f1[i]] = sum1[i].get(f1[i], 0) + 1 / len(feature1)
        self.condition_prob[0]=sum0;
        self.condition_prob[1]=sum1;




        # ********* End *********#

    def predict(self, feature):
        '''
        对数据进行预测，返回预测结果
        :param feature:测试数据集所有特征组成的ndarray
        :return:
        '''
        # ********* Begin *********#
        result=[]
        for f in feature:
            result1=self.label_prob[1]
            result0=self.label_prob[0]
            for i in range(len(f)):
                result0*=self.condition_prob[0][i].get(f[i],0)
                result1*=self.condition_prob[1][i].get(f[i],0)
            if result0>result1:
                result.append(0)
            else:
                result.append(1)
        return np.array(result)
        # ********* End *********#
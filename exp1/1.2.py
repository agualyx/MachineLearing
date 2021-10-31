from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

def classification(train_feature, train_label, test_feature):
    '''
    对test_feature进行红酒分类
    :param train_feature: 训练集数据，类型为ndarray
    :param train_label: 训练集标签，类型为ndarray
    :param test_feature: 测试集数据，类型为ndarray
    :return: 测试集数据的分类结果
    '''

    #********* Begin *********#
    #使用StandardScaler对测试集进行标准化
    scaler=StandardScaler()
    std_train_feature=scaler.fit_transform(train_feature)
    #使用KNeighborsClassifier预测数据并返回训练后的数据
    classifier=KNeighborsClassifier()
    classifier.fit(std_train_feature,train_label)
    return classifier.predict(scaler.fit_transform(test_feature))
    #********* End **********#

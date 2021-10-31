from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine

def scaler(data):
    '''
    返回标准化后的红酒数据
    :param data: 红酒数据对象
    :return: 标准化后的红酒数据，类型为ndarray
    '''

    #********* Begin *********#
    scaler=StandardScaler()
    return scaler.fit_transform(data['data'])

    #********* End **********#
data=load_wine()
print(scaler(data))
import numpy as np
from sklearn.datasets import load_wine
def alcohol_mean(data):
    '''
    返回红酒数据中红酒的酒精平均含量
    :param data: 红酒数据对象
    :return: 酒精平均含量，类型为float
    '''

    #********* Begin *********#
    wine_data=data['data']
    data_mean=wine_data.mean(0)
    return data_mean[0]
    #********* End **********#

data=load_wine()
avg=alcohol_mean(data)
print(avg)

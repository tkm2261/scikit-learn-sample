#coding:utf-8

import numpy
import sys
import pickle
from sklearn.linear_model import Ridge
from sklearn.cross_validation import cross_val_score

DELIMITOR=","


def load_data(data_path="data/YearPredictionMSD.txt"):
    '''
    データの読込
    YearPredictionMSD
    https://archive.ics.uci.edu/ml/datasets/YearPredictionMSD
    '''
    
    f = open(	data_path)
    list_data = []
    list_target = []

    for line in f:
        line = line.strip("\n").split(",")
        data = line[1:]
        target = float(line[0])

        data = [float(ele) for ele in data]

        list_data.append(data)
        list_target.append(target)

    f.close()

    return list_target, list_data
    
    
def read_data_with_numpy(data_path="data/YearPredictionMSD.txt"):
    '''
    numpyを使った読込
    YearPredictionMSD
    https://archive.ics.uci.edu/ml/datasets/YearPredictionMSD
    '''
    data = numpy.loadtxt(data_path, delimiter=DELIMITOR)
    target = data[:, 0]
    data = data[:, 1:]

    return target, data
    

    
def ridge_tuning_and_training(data, target, cv_number=10):
    '''
    リッジ回帰のグリッドサーチ＋CV＋最良パラメータでの学習
    '''
    
    max_score = -1 * sys.maxint
    max_alpha = 0
    
    #グリッドサーチ
    for alpha in [10**i for i in xrange(-4,5)]:
    
        #リッジ回帰のインスタンス生成
        model = Ridge(alpha=alpha)
        
        #交差検定 n_jobs=-1でマルチコア
        cv_scores = cross_val_score(model,
                           data,
                           target,
                           cv=cv_number,
                           scoring='mean_squared_error',
                           n_jobs=-1)
                           
        #交差検定の各学習の平均値計算
        score = numpy.mean(cv_scores)
        
        print "alpha:{} score:{}".format(alpha, score)
        
        #スコアが良かったらパラメタを記憶
        if score > max_score:
            max_score = score
            max_alpha = alpha
            
    print "best_alpha:{} best_score:{}".format(max_alpha, max_score)
    #最良パラメータで学習
    model = Ridge(alpha=max_alpha)
    model.fit(data, target)
    
    return model
    
if __name__ == "__main__":

    target, data = load_data()
    ridge_model = ridge_tuning_and_training(data, target, cv_number=10)
    
    # シリアライズしておくと後々予測で使える。
    with open("ridge_model", "w") as f:
        pickle.dump(ridge_model)
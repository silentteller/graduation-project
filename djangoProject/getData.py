from django.http import JsonResponse
import json
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.inspection import partial_dependence
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
from sklearn import metrics
from treeinterpreter import treeinterpreter as ti
from sklearn.inspection import plot_partial_dependence

feature = None
mdiFeature = None
pdpData = []
estimator = None
totaldata = []


def get(request):
    filename = "titanic_train.csv"
    treeNum = 0
    treeDeep = 0
    if request.method == 'GET':
        filename = request.GET.get('name')
        treeNum = request.GET.get('treeNum')
        treeDeep = request.GET.get('treeDeep')
        print(filename + " " + treeNum + " " + treeDeep)
    # D:\PyCharm 2020.2.3\djangoProject\djangoProject\data
    data = pd.read_csv('D:/PyCharm 2020.2.3/djangoProject/djangoProject/data/' + filename + '1.csv')
    # dftrain = pd.read_csv('data/' + filename)
    y_train = None
    dp = None
    if filename == 'german':
        y_train = data.pop('Creditability')
        dp = pd.read_csv('D:/PyCharm 2020.2.3/djangoProject/djangoProject/data/germandis.csv')
    else:
        y_train = data.pop('survived')
    # 定义随机森林模型参数
    rfc = RandomForestClassifier(max_depth=int(treeDeep), n_estimators=int(treeNum), random_state=60)
    # # 处理数据，将离散化的值转换为数字等
    # data = prepareData(dftrain)

    global totaldata
    totaldata = data.copy()
    # 特征名称列表
    featureList = totaldata.columns.values.tolist()
    # # 训练
    rfc.fit(data, y_train)

    global estimator
    estimator = rfc

    importancenData = permutation_importance(rfc, data, y_train, n_repeats=100, random_state=16)

    global mdiFeature
    mdiFeature = rfc.feature_importances_.tolist()

    global feature
    feature = importancenData.importances_mean.tolist()
    print("排列重要性")
    print(feature)
    print("mdi重要性")
    print(mdiFeature)

    # 计算部份依赖
    global pdpData
    pdpData = []
    for index, value in enumerate(data.columns.values):
        pdp, axes = partial_dependence(rfc, data, index)
        pdpData.append({"name": value, "axes": axes[0].tolist(), "pdp": pdp[0].tolist()})
    # print(plot_partial_dependence(rfc, data, ["Account Balance"], target=0))
    # # tsne降维
    # tsne = TSNE(n_components=2, perplexity=30, n_iter=500, metric='precomputed')

    tsne = TSNE(learning_rate=100.0)
    array = tsne.fit_transform(dp).tolist()  # 进行数据降维

    # embedding = MDS(n_components=2)
    # array = embedding.fit_transform(dp)

    # 预测的概率
    y_pre = rfc.predict(data)
    predict_prob = rfc.predict_proba(data)
    predict0 = []
    predict1 = []
    for i in predict_prob:
        predict0.append(i[0])
        predict1.append(i[1])
    data['predict0'] = predict0
    data['predict1'] = predict1
    # # 将预测值和真实值加入到数据中
    data['predict'] = y_pre
    data['true'] = y_train
    x = []
    y = []
    for i in array:
        x.append(i[0])
        y.append(i[1])
    data['x'] = x
    data['y'] = y
    index = []
    for i in range(len(data)):
        index.append(i)
    data.insert(0, 'id', index)
    # data = pd.read_csv('D:/PyCharm 2020.2.3/djangoProject/djangoProject/data/result_new.csv')
    # return JsonResponse(data, safe=False)

    print("AUC Score (Train): %f" % metrics.roc_auc_score(y_train, y_pre))
    print(metrics.confusion_matrix(y_train, y_pre, labels=None, sample_weight=None))
    da = data.to_dict(orient='records')
    featureListMin = []
    featureListMax = []
    for i in featureList:
        featureListMin.append(min(data[i]))
        featureListMax.append(max(data[i]))
    return JsonResponse({'data': da, 'featureList': featureList, 'featureImportance': feature,
                         'mdiFeatureImportance': mdiFeature, 'featureListMin': featureListMin
                         , 'featureListMax': featureListMax, 'auc': metrics.roc_auc_score(y_train, y_pre),
                         'confusionMatrix': metrics.confusion_matrix(y_train, y_pre, labels=None, sample_weight=None).tolist()}, safe=False)


def getCharacterInfo(request):
    print(pdpData)
    return JsonResponse(pdpData, safe=False)


def getDPath(request):
    id = -1
    if request.method == 'GET':
        id = int(request.GET.get('id'))
    # 特征名称列表
    featureList = totaldata.columns.values.tolist()
    # 选择的数据，由ndarray转换成二维list
    selData = totaldata.iloc[id].values.reshape((1, len(totaldata.columns.values.tolist()))).tolist()
    # print(selData)
    fmin = []
    fmax = []
    for i in featureList:
        fmin.append(float(totaldata[i].min()))
        fmax.append(float(totaldata[i].max()))
    # 随机森林中的决策树集合, estimator.estimators_是一个list, 随机森林中的决策树，tree是其存储的结构体estimator.estimators_[0].tree_
    # p保存对应的决策路径，决策树路径提取0->[{特征，值，符号},{...}],[5, 1.5, <=]
    # rule保存路径提取的规则，特征->[min, max], predict每条决策路径预测的最终概率向量
    p = {}
    rule = {}
    predict = []
    for j in range(len(estimator.estimators_)):
        m = estimator.estimators_[j].decision_path(selData)
        predict.append(estimator.estimators_[j].predict_proba(selData)[0].tolist())
        d = []
        r = {}  # 决策路径和其对应的特征，范围值，0:[-1.5, 2]
        for i in m[0].indices:
            temp = []
            f = int(estimator.estimators_[j].tree_.feature[i])  # 特征index
            threshold = estimator.estimators_[j].tree_.threshold[i]  # 节点分裂值
            if (estimator.estimators_[j].tree_.feature[i] >= 0):
                if f not in r.keys():
                    r[f] = [fmin[f], fmax[f]]
                temp.append(int(f))
                temp.append(float(threshold))
                if (selData[0][f] <= threshold):
                    temp.append("<=")
                    if threshold < r[f][1]:
                        r[f][1] = float(threshold)
                else:
                    temp.append(">")
                    if threshold > r[f][0]:
                        r[f][0] = float(threshold)
                d.append(temp)
        p[j] = d
        rule[j] = r

    # compute Decision Path Distance
    dp = np.zeros([len(estimator.estimators_), len(estimator.estimators_)])
    for i in range(len(estimator.estimators_)):
        for j in range(len(estimator.estimators_)):
            # print(str(i) + " " + str(j))
            if (i == j):
                continue
            dist = 0
            featureCount = 0
            for k in range(len(featureList)):
                if (k not in rule[i].keys() and k not in rule[j].keys()):
                    continue
                elif (k not in rule[i].keys() or k not in rule[j].keys()):
                    dist += 1
                    featureCount += 1
                else:
                    dist += (abs(rule[i][k][0] - rule[j][k][0]) + abs(rule[i][k][1] - rule[j][k][1])) / (
                            fmax[k] - fmin[k]) / 2
                    featureCount += 1
            if (featureCount != 0):
                dp[i][j] = dist / featureCount
    # print(dp)
    tsne = TSNE(learning_rate=50.0)
    # tsne = TSNE(learning_rate=100.0)
    array = tsne.fit_transform(dp).tolist()  # 进行数据降维

    #contribution compute
    prediction, bias, contributions = ti.predict(estimator, totaldata.iloc[id].values.reshape((1, len(totaldata.columns.values.tolist()))))
    # print(prediction, bias, contributions)

    return JsonResponse({'path': p, 'rule': rule, 'predict': predict, 'array': array, 'fmin': fmin, 'fmax': fmax,
                         'featureList': featureList, 'data': totaldata.iloc[id].tolist(), 'featureImportant': feature,
                         'prediction': prediction.tolist(), 'bias': bias.tolist(), 'contributions': contributions.tolist()}, safe=False)


def rePredict(request):
    fixData = []
    temp = []
    # 特征名称列表
    featureList = totaldata.columns.values.tolist()
    # print(selData)
    fmin = []
    fmax = []
    for i in featureList:
        fmin.append(float(totaldata[i].min()))
        fmax.append(float(totaldata[i].max()))
    for i in featureList:
        temp.append(float(request.GET.get(i)))
    fixData.append(temp);
    p = {}
    rule = {}
    predict = []
    result = estimator.predict_proba(fixData).tolist()
    for j in range(len(estimator.estimators_)):
        m = estimator.estimators_[j].decision_path(fixData)
        predict.append(estimator.estimators_[j].predict_proba(fixData)[0].tolist())
        d = []
        r = {}  # 决策路径和其对应的特征，范围值，0:[-1.5, 2]
        for i in m[0].indices:
            temp = []
            f = int(estimator.estimators_[j].tree_.feature[i])  # 特征index
            threshold = estimator.estimators_[j].tree_.threshold[i]  # 节点分裂值
            if (estimator.estimators_[j].tree_.feature[i] >= 0):
                if f not in r.keys():
                    r[f] = [fmin[f], fmax[f]]
                temp.append(int(f))
                temp.append(float(threshold))
                if (fixData[0][f] <= threshold):
                    temp.append("<=")
                    if threshold < r[f][1]:
                        r[f][1] = float(threshold)
                else:
                    temp.append(">")
                    if threshold > r[f][0]:
                        r[f][0] = float(threshold)
                d.append(temp)
        p[j] = d
        rule[j] = r
    # contribution compute
    fixDataNDArray = np.array(fixData)
    prediction, bias, contributions = ti.predict(estimator, fixDataNDArray)
    print(prediction, bias, contributions)
    return JsonResponse({'path': p, 'rule': rule, 'predict': predict, 'featureList': featureList, 'result': result, 'data': fixData[0],
                         'prediction': prediction.tolist(), 'bias': bias.tolist(), 'contributions': contributions.tolist()},
                        safe=False)


def prepareData(df_data):
    # df=df_data.drop(['name'],axis=1)#名字训练时不需要，去掉
    df = df_data.drop(['deck'], axis=1)
    age_mean = df['age'].mean()
    df['age'] = df['age'].fillna(age_mean)  # 缺失的年龄以平均值填充
    fare_mean = df['fare'].mean()
    df['fare'] = df['fare'].fillna(fare_mean)  # 缺失的票价以平均值填充
    df['sex'] = df['sex'].map({'female': 0, 'male': 1}).astype(int)  # 文字转化为数字表示
    df['embark_town'] = df['embark_town'].fillna('Southampton')  # 缺失值用最多的值取代
    dict = {}
    for key in df_data['embark_town']:
        dict[key] = dict.get(key, 0) + 1
    print(dict)
    df['embark_town'] = df['embark_town'].map({'Cherbourg': 0, 'Queenstown': 1, 'Southampton': 2, 'unknown': 2}).astype(
        int)  # 文字转化为数字表示
    df['alone'] = df['alone'].map({'y': 1, 'n': 0}).astype(int)
    df['class'] = df['class'].map({'First': 0, 'Second': 1, 'Third': 2}).astype(int)
    # ndarray_data=df.values
    # features=ndarray_data[:,1:]#没有生存情况
    # label=ndarray_data[:,0]#生存情况
    # minmax_scale=preprocessing.MinMaxScaler(feature_range=(0,1))
    # norm_features=minmax_scale.fit_transform(features)#归一化
    return df


def computeDistance(data):
    dp = np.zeros([len(data), len(data)])
    maxAge = max(data['age'])
    minAge = min(data['age'])
    maxFare = max(data['fare'])
    minFare = min(data['fare'])
    for i in range(len(data)):
        for j in range(len(data)):
            print(str(i) + " " + str(j))
            dis = 0
            # sex
            if data.iloc[i]['sex'] != data.iloc[j]['sex']:
                dis += 1
            # age
            dis += abs(data.iloc[i]['age'] - data.iloc[j]['age']) / (maxAge - minAge)
            # n_siblings_spouses
            if data.iloc[i]['n_siblings_spouses'] != data.iloc[j]['n_siblings_spouses']:
                dis += 1
            # parch
            if data.iloc[i]['parch'] != data.iloc[j]['parch']:
                dis += 1
            # fare
            dis += abs(data.iloc[i]['fare'] - data.iloc[j]['fare']) / (maxFare - minFare)
            # class
            if data.iloc[i]['class'] != data.iloc[j]['class']:
                dis += 1
            # embark_town
            if data.iloc[i]['embark_town'] != data.iloc[j]['embark_town']:
                dis += 1
            # alone
            if data.iloc[i]['alone'] != data.iloc[j]['alone']:
                dis += 1
            dp[i][j] = dis
    return dp

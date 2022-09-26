import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from pylab import savefig
from utility import  Utilty
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import mean_squared_error
from sklearn.svm import OneClassSVM
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import numpy as np

period_size = 40
step_size = 5

leaf_nodes=[]
child_tree = []




def find_ma(ts):
    weights = np.arange(1, period_size+2)
    wma10 = ts.rolling(period_size+1).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)
   # print(wma10)
    return wma10
def find_ewa(ts,rolmean):
    newts=ts.copy()
    newts.iloc[0:period_size+1] = rolmean[0:period_size+1]
   # ewa = ts.ewm(span=12).mean()
    ewa=newts.ewm(span=period_size+1, adjust=False).mean()
    #print(ewa)
    return ewa
def find_twma(ts):
    weights = np.arange(1, 4)
    avg = ts.rolling(3).mean()
    avg = avg.dropna()
    new_res = avg.rolling(3).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)
   # print(new_res)
    return new_res
def find_latesttwma(ts):
    twma_res = ts.copy()
    twma_res = twma_res.iloc[period_size:]
    steps = step_size
    i = 0
    j = period_size
    finalres = []
    while j <= len(ts):
        newres = []
        #weights = np.arange(1, 5)
        weights = []

        temp=ts[i:j]
       # print(temp)

        multiplier_weight=1
        for m in range(0, len(temp), steps):
            avg = temp[m:m+steps].sum()/steps
            weights.append(temp[m:m+steps].std()**multiplier_weight)
            #weights.append(multiplier_weight)
            multiplier_weight = multiplier_weight+1
            newres.append(avg)
        #newres = pd.DataFrame(newres)
        weights = np.array(weights)
        estimate = np.multiply(newres , weights)
        estimate = estimate.sum()/ weights.sum()
       # estimate = newres.apply(lambda prices: np.dot(prices, weights) / weights.sum())
        finalres.append(estimate)
        j = j+1
        i = i+1
   # print('lenght of finalres ',len(finalres))
    for i in range(len(finalres)-1):
        twma_res.iloc[i] = finalres[i]
    #twma_res.iloc[0:len(finalres)]=finalres.iloc[0:len(finalres)]
    return twma_res

def find_wma(ts):
    twma_res = ts.copy()
    twma_res = twma_res.iloc[period_size:]
    steps = step_size
    i = 0
    j = period_size
    finalres = []
    while j <= len(ts):
        newres = []
        # weights = np.arange(1, 5)
        weights = []

        temp = ts[i:j]
       # print(temp)

        multiplier_weight = 1
        for m in range(0, len(temp),1):
            #avg = temp[m:m + steps].sum() / steps

            weights.append(multiplier_weight)
            # weights.append(multiplier_weight)
            multiplier_weight = multiplier_weight + 1
            newres.append(temp[m])
        # newres = pd.DataFrame(newres)
        weights = np.array(weights)
        estimate = np.multiply(newres, weights)
        estimate = estimate.sum() / weights.sum()
        # estimate = newres.apply(lambda prices: np.dot(prices, weights) / weights.sum())
        finalres.append(estimate)
        j = j + 1
        i = i + 1
   # print('lenght of finalres ', len(finalres))
    for i in range(len(finalres) - 1):
        twma_res.iloc[i] = finalres[i]
    # twma_res.iloc[0:len(finalres)]=finalres.iloc[0:len(finalres)]
    return twma_res

def find_avgstdtwma(ts):
    twma_res = ts.copy()
    twma_res = twma_res.iloc[period_size:]
    steps = step_size
    i = 0
    j = period_size
    positive = True
    finalres = []
    while j <= len(ts):
        newres = []
        std_array=[]
        #weights = np.arange(1, 5)
        weights = []

        temp=ts[i:j]
       # print(temp)

        multiplier_weight=1
        for m in range(0, len(temp), steps):
            avg = temp[m:m+steps].sum()/steps
            std1 = temp[m:m+steps].std()
            std_array.append(std1)
            #weights.append(temp[m:m+steps].std()*multiplier_weight)
            weights.append(multiplier_weight)
            multiplier_weight = multiplier_weight+1
            newres.append(avg)
            if m+steps == len(temp):
                if avg < temp[m+steps-1]:
                    positive = True
                else:
                    positive = False
        #newres = pd.DataFrame(newres)
        weights = np.array(weights)
        estimate = np.multiply(newres, weights)
        std_estimate = np.multiply(std_array, weights)
        estimate = estimate.sum() / weights.sum()
        std_estimate = std_estimate.sum() / weights.sum()
        if positive:
            estimate = estimate + std_estimate*0.001
        else:
            estimate = estimate - std_estimate*0.001
       # estimate = newres.apply(lambda prices: np.dot(prices, weights) / weights.sum())
        finalres.append(estimate)
        j = j+1
        i = i+1
   # print('lenght of finalres ',len(finalres))
    for i in range(len(finalres)-1):
        twma_res.iloc[i] = finalres[i]
    #twma_res.iloc[0:len(finalres)]=finalres.iloc[0:len(finalres)]
    return twma_res

if __name__ == '__main__':
    data = pd.read_csv('aws5f5533_data.csv')
    #data = pd.read_csv('Shampoo.csv')
    print(data.head())
    print('\n Data Types:')
    print(data.dtypes)
    #con = data['TravelDate']
    data['Date'] = pd.to_datetime(data['Date'])
    #data['Month'] = pd.to_datetime(data['Month'])
    data.set_index('Date', inplace=True)
    #data.set_index('Month', inplace=True)
    print(data.index)
    ts = data['Value']
    #ts = data['Sal
   # ts=ts.diff()
   # ts=ts.diff()
    #ts = np.log(ts)
    print(ts)
    pd.plotting.register_matplotlib_converters()
   # plt.plot(ts)
    #plt.show()
    rolmean = ts.rolling(period_size+1).mean()
    print('mean', rolmean)
    twma = find_avgstdtwma(ts)
    ewa = find_ewa(ts,rolmean)
    wma = find_wma(ts)

    twma = pd.DataFrame(twma)
    newdata={}
    newdata['twma'] = np.round(twma, decimals=3)
    newdata['wma'] = np.round(wma, decimals=3)
    newdata['ewa'] = np.round(ewa, decimals=3)
    print("******************************************************")
    #print(newdata)
    #ts['TravelDate', 'twma', 'wma'].head(20)

    ts=ts.iloc[period_size:]
   # newdata['ewa'] = newdata['ewa'].iloc[12:]

    newdata['twma'] = newdata['twma'].dropna()
    newdata['wma'] = newdata['wma'].dropna()
    newdata['ewa'] = newdata['ewa'].dropna()
    rolmean = rolmean.dropna()
   # newdata['twma'] = newdata['twma'][:-1]
    #newdata['twma'] = newdata['twma'].iloc[1:]
    #newdata['ewa'] = newdata['ewa'].iloc

    rs_twma= np.sqrt(mean_squared_error(ts,newdata['twma']))
    rs_wma=np.sqrt(mean_squared_error(ts,newdata['wma']))
    rs_ewa= np.sqrt(mean_squared_error(ts,newdata['ewa']))
    rs_ma = np.sqrt(mean_squared_error(ts, rolmean))
    twma = newdata['twma']

    error = abs(twma['Value'] - ts)
#    print('error', error[1273])
  #  print('error', error[1274])
    print('rs_twma',rs_twma)
    print('rs_wma',rs_wma)
    print('rs_ewa',rs_ewa)
    #print('error',twma)

    #plt.show()
    result =np.array(error).reshape(-1, 1)

    y_true = np.zeros(1399)  # ground truth labels
    y_true[1391] = 1
    y_true[1392] = 1
    y_true[1393] = 1
    y_true[1394] = 1
    y_true[1395] = 1
    y_true[1396] = 1
    y_true[1397] = 1
    y_true[1398] = 1

    y_btai = np.zeros(1399)  # ground truth labels
    y_btai[1391] = 1
    y_btai[1392] = 1
    y_btai[1393] = 1
    y_btai[1394] = 1
    y_btai[1395] = 1
    y_btai[1396] = 1
    y_btai[1397] = 1
    y_btai[1298] = 1
    util = Utilty()
    rng = np.random.RandomState(42)


    X_train=[]
    train_threshold = math.floor(0.7 * len(result))
    for i in range(train_threshold):
        X_train.append(result[i])
    # Generating new, 'normal' observation
    X_test = result
    #X_test = np.r_[X_test + 3, X_test]
    #X_test = pd.DataFrame(X_test, columns=['x1', 'x2'])

    # Generating outliers
    #X_outliers = rng.uniform(low=-1, high=5, size=(5, 2))
    #X_outliers = pd.DataFrame(X_outliers, columns=['x1', 'x2'])
    #clf = IsolationForest(max_samples=40)
    random_state = np.random.RandomState(42)
    clf = IsolationForest(n_estimators=100,max_samples='auto',contamination=float(0.2),random_state=random_state)
    clf.fit(X_train)

    # predictions
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    #y_pred_outliers = clf.predict(X_outliers)
    # new, 'normal' observations ----
    print("Accuracy:", list(y_pred_test).count(1) / y_pred_test.shape[0])
    for itr in range(len(y_pred_test)):
        if y_pred_test[itr]==1:
            y_pred_test[itr]=0
        else:
            y_pred_test[itr] = 1
        print('iforestanomaly',y_pred_test[itr],'data',result[itr])
    fpr1, tpr1, threshold = metrics.roc_curve(y_true, y_pred_test)
    roc_auc_if = metrics.auc(fpr1, tpr1)
    X_test = result
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    cov = EllipticEnvelope().fit(X_train)
    # Now we can use predict method. It will return 1 for an inlier and -1 for an outlier.
    y_pred_test = cov.predict(X_test)
    for itr in range(len(y_pred_test)):
        print('anomaly envelope', y_pred_test[itr], 'data', X_test[itr])

    for itr in range(len(y_pred_test)):
        if y_pred_test[itr] == 1:
            y_pred_test[itr] = 0
        else:
            y_pred_test[itr] = 1
    fpr2, tpr2, threshold = metrics.roc_curve(y_true, y_pred_test)
    roc_auc_env = metrics.auc(fpr2, tpr2)
    X_test = result
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    OSVMclf = OneClassSVM().fit(X_train)
    OSVMclf.predict(X_test)
    # Now we can use predict method. It will return 1 for an inlier and -1 for an outlier.

    y_pred_test = OSVMclf.predict(X_test)
    for itr in range(len(y_pred_test)):
        print('anomaly ocsvm', y_pred_test[itr], 'data', X_test[itr])
    for itr in range(len(y_pred_test)):
        if y_pred_test[itr] == 1:
            y_pred_test[itr] = 0
        else:
            y_pred_test[itr] = 1
    fpr3, tpr3, threshold = metrics.roc_curve(y_true, y_pred_test)
    roc_auc_svm = metrics.auc(fpr3, tpr3)
    # Accuracy: 0.93
    # outliers ----
    #print("Accuracy:", list(y_pred_outliers).count(-1) / y_pred_outliers.shape[0])
    # Accuracy: 0.96






    y_probas = y_btai  # predicted probabilities generated by sklearn classifier
    y_probas1 = y_pred_test
    fpr, tpr, threshold = metrics.roc_curve(y_true, y_probas)
    roc_auc = metrics.auc(fpr, tpr)


    # skplt.metrics.plot_roc_curve(y_true, y_probas)
    # plt.show()
    plt.title('Receiver Operating Characteristic', fontsize=20)
    plt.plot(fpr, tpr, color="red",linewidth=4, label='WTWMEMGBTAI AUC = %0.2f' % roc_auc)
    plt.plot(fpr1, tpr1, 'b', linewidth=4, label='iForest AUC = %0.2f' % roc_auc_if)
    plt.plot(fpr2, tpr2, 'g', linewidth=4, label='Envelope AUC = %0.2f' % roc_auc_env)
    plt.plot(fpr3, tpr3, 'orange', linewidth=4, label='OCSVM AUC = %0.2f' % roc_auc_svm)
    #plt.plot(fpr, tpr, linestyle='solid', label='No Skill')
    #plt.plot(fpr1, tpr1, marker='.', label='Logistic')
    plt.legend(loc='lower right', prop={'size': 14})
   # plt.plot([0, 1], [0, 1], 'r--')
    #plt.xlim([0, 1])
    #plt.ylim([0, 1])
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.show()

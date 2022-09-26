import numpy as np
import pandas as pd
import math
from sklearn.covariance import EllipticEnvelope
import time


if __name__ == '__main__':
    start_time = time.time()
    data = pd.read_csv('aws5f5533_data.csv')
    # data = pd.read_csv('Shampoo.csv')
    print(data.head())
    print('\n Data Types:')
    print(data.dtypes)
    # con = data['TravelDate']
    data['Date'] = pd.to_datetime(data['Date'])
    # data['Month'] = pd.to_datetime(data['Month'])
    data.set_index('Date', inplace=True)
    # data.set_index('Month', inplace=True)
    print(data.index)
    ts = data['Value']
    # ts = data['Sal
    # ts=ts.diff()
    # ts=ts.diff()
    # ts = np.log(ts)
    print(ts)


    # plt.show()
    result = ts
    X_train = []
    train_threshold = math.floor(0.7 * len(result))
    for i in range(train_threshold):
        X_train.append(result[i])
    # Generating new, 'normal' observation
    X_test = result
    X_train = np.array(X_train).reshape(-1, 1)
    X_test = np.array(X_test).reshape(-1, 1)
    cov = EllipticEnvelope().fit(X_train)
    # Now we can use predict method. It will return 1 for an inlier and -1 for an outlier.
    y_pred_test=cov.predict(X_test)
    for itr in range(len(y_pred_test)):
        print('anomaly', y_pred_test[itr], 'data', X_test[itr])
    print("--- %s seconds ---" % (time.time() - start_time))
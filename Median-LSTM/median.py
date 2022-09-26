import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
from tensorflow.keras.layers import Activation
from tensorflow.keras import backend as K
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras import losses
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Layer
import tensorflow as tf
from tensorflow.keras.models import Model

from keract import get_activations, persist_to_json_file, load_activations_from_json_file
period_size = 76
step_size = 38
upper_q_threshold=0.5
lower_q_threshold=0.0001
output = 'activations.json'
path='final.csv'
first_layer=[]
second_layer=[]
third_layer=[]
fourth_layer=[]
swish_layer=[]
anomaly_res =[-563.9761436,-2416.661669,-3232.353099,-3565.527389,-2560.50266,-4133.645629,-4004.058979,-4279.506572,-4804.726653,-5948.068253]
alpha_values = list()
class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.beta = tf.Variable(0.1)


        #self.d0 = Dense(16, activation=self.prelu)
        #self.d1 = Dense(32, activation=self.prelu)
        #self.d2 = Dense(3, activation='softmax')
        #modelq10 = Sequential()
        self.d0=LSTM(4, input_shape=(multi, 1),activation=self.prelu)
        self.d1 = Dense(32, activation=self.prelu)
        #modelq10.add(LSTM(4, input_shape=(multi, 1)))
        #modelq10.add(Dense(1))
        #modelq10.add(Swish(beta=1.5))

    def prelu(self, x):
        #print('beta_prelu',self.beta)
        return K.softsign(x) * self.beta
        #return tf.maximum(self.min_value, x * self.prelu_slope)

    def call(self, x, **kwargs):
        print('beta',self.beta)
        alpha_values.append(self.beta)

        x = self.d0(x)
        x = self.d1(x)

        return x


class Swish(Layer):
    def __init__(self, beta, **kwargs):
        super(Swish, self).__init__(**kwargs)
        self.beta = K.cast_to_floatx(beta)
        self.beta=tf.Variable(0.1)
        print('beta',self.beta)

    def call(self, inputs):
        print('beta', self.beta,'input',input)
        return K.softsign(inputs) * self.beta

    def get_config(self):
        config = {'beta': float(self.beta)}
        base_config = super(Swish, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    def get_weights(self):

        base_config = super(Swish, self).get_weights()
        return dict(list(base_config.items()))
    def compute_output_shape(self, input_shape):
        return input_shape

class LogThirdLayerOutput(Callback):
    def on_epoch_end(self, epoch, logs=None):
        '''print(self.model.layers[2].output)
        get_3rd_layer_output = K.function([self.model.layers[0].input],
                                          [self.model.layers[2].output])
        print(get_3rd_layer_output(self.validation_data))
        layer_output = get_3rd_layer_output(self.validation_data)[0]'''
        outputs = []
        x = numpy.ones((1,2, 1))
        activations = get_activations(self.model, x)
        #print('weight',self.model.su)
        print('lstm weight',self.model.layers[0].get_weights()[0])
        print('config',self.model.layers[0].get_config())
        print('dense weight', self.model.layers[1].get_weights()[0])
        print('pef weight', self.model.layers[2].get_weights())
        row=''
        for k, v in activations.items():
            print('key',k,'value', v.tolist())
            if k=='lstm':
               lstm_array= v.tolist()
               lstm_array=lstm_array[0]
               for i in range(len(lstm_array)):
                    row=row+","+str(lstm_array[i])
                    if i==0:
                        first_layer.append(lstm_array[i])
                    elif i==1:
                        second_layer.append(lstm_array[i])
                    elif i==2:
                        third_layer.append(lstm_array[i])
                    else:
                        fourth_layer.append(lstm_array[i])
            if k=='swish_1':
                swish_array = v.tolist()
                swish_layer.append(swish_array[0])
        row=row+"\n"
        csv_file = open(path, 'a')
        csv_file.write(row)

        # persist the activations to the disk.

        persist_to_json_file(activations, output)

        # read them from the disk.
        activations2 = load_activations_from_json_file(output)

        # print them.
        print(list(activations.keys()))
        print(list(activations2.keys()))
        print('Dumped to {}.'.format(output))
        '''for layer in self.model.layers:
            keras_function = K.function([self.model.input], [layer.output])
            outputs.append(keras_function([self.validation_data, 1]))
        print(outputs)'''
        #print(layer_output.shape)
def intersection(lst1, lst2):
    lst3=[]
    for itr in range(len(lst1)):
        if lst1[itr][0] in lst2:
            lst3.append(lst1[itr])
    return lst3

def custom_activation(x):
    #beta=K.variable(alpha)
    a=K.softsign(x)
    return a

get_custom_objects().update({'custom_activation': Swish(beta=1.5)})

def verify_stationarity(dataset):
    is_stationary=True
    test_results = adfuller(dataset)

    print(f"ADF test statistic: {test_results[0]}")
    print(f"p-value: {test_results[1]}")
    print("Critical thresholds:")

    for key, value in test_results[4].items():
        print(f"\t{key}: {value}")
    itr = 0
    for key, value in test_results[4].items():
       print('\t%s: %.3f' % (key, value))
       if itr==0:
         critical=value
       itr=itr+1

    print('critical',critical)
    if test_results[0] > critical:
         print('non stationary')
         is_stationary=False
    return  is_stationary

def create_dataset(dataset, look_back=1, tw=3):
    dataX, dataY = [], []  # dtaset for mean
    datastdX, datastdY = [], []  # dataset for std
    datacombX, datacomY = [], []  # dataset for mean and std for third deep learning
    multi = look_back // tw
    for i in range(len(dataset) - look_back - 1):
        q50X = []
        a = dataset[i + 1:(i + look_back + 1)]
        indices = i + (multi - 1) * tw
        # print('last window', dataset[indices:(i + look_back), 0])
        c = numpy.quantile(a, upper_q_threshold)
        for j in range(0, len(a), tw):
            q50 = numpy.quantile(a[j:j + tw], upper_q_threshold)
            q50X.append(q50)
        dataX.append(q50X)
        dataY.append(c)

    return numpy.array(dataX), numpy.array(dataY)
def identify_anomaly_quantiles(prediction_errors):
    anomaly_detection=[]
    for m in range(0, len(prediction_errors), period_size):
        period_prediction_errors=prediction_errors[m:m + period_size]
        upper_threshold = numpy.quantile(prediction_errors[m:m + period_size],0.9)
        lower_threshold = numpy.quantile(prediction_errors[m:m + period_size],0.1)
        #upper_threshold=avg+2*std1
        #lower_threshold = avg - 2 * std1
        for i in range(len(period_prediction_errors)):
            if (period_prediction_errors[i]>0 and period_prediction_errors[i]> upper_threshold) or (period_prediction_errors[i]<0 and period_prediction_errors[i]< lower_threshold):
                anomaly_detection.append(period_prediction_errors[i])

    return anomaly_detection

def identify_anomaly(prediction_errors):
    anomaly_detection=[]
    for m in range(0, len(prediction_errors), period_size):
        period_prediction_errors=prediction_errors[m:m + period_size]
        avg = numpy.average(prediction_errors[m:m + period_size])
        std1 = numpy.std(prediction_errors[m:m + period_size])
        upper_threshold=avg+2*std1
        lower_threshold = avg - 2 * std1
        for i in range(len(period_prediction_errors)):
            if (period_prediction_errors[i]> upper_threshold) or ( period_prediction_errors[i]< lower_threshold):
                anomaly_detection.append(period_prediction_errors[i])

    return  anomaly_detection
if __name__ == '__main__':
    # fix random seed for reproducibility
    numpy.random.seed(7)
    # load the dataset
    dataframe = read_csv('/content/aws5f5533_data_normal.csv', usecols=[1], engine='python')
    dataset = dataframe.values
    #stationary=verify_stationarity(dataset)
    #dataframe=dataframe.diff(axis = 0, periods = 1)
    #dataset = dataframe.dropna().values
    dataset = dataset.astype('float32')
    # normalize the dataset
    print('dataset', dataset)
    #dataset=dataset.dropna()
    #stationary = verify_stationarity(dataset)
    scaler = MinMaxScaler(feature_range=(0, 1))

    dataset = scaler.fit_transform(dataset)
    # split into train and test sets
    train_size = int(len(dataset) * 0.7)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    # reshape into X=t and Y=t+1
    look_back = period_size
    tw = step_size
    multi = look_back // tw
    trainX, trainY = create_dataset(train, look_back, tw)
    testX, testY = create_dataset(test, look_back, tw)
    print(trainX)
    # reshape input to be [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))
    print(trainX)

    '''modelq10 = Sequential()
    modelq10.add(LSTM(4, input_shape=(multi, 1)))
    modelq10.add(Dense(1))
    modelq10.add(Swish(beta=1.5))'''
    #modelq10.add(Activation(Swish(beta=1.5)))
    #modelq10.add(Activation(custom_activation, name='SpecialActivation'))
    modelq10=MyModel()
    modelq10.compile(loss=losses.logcosh, optimizer='adam',run_eagerly=True)

    modelq10.fit(trainX, trainY, epochs=10, batch_size=1, verbose=2)
    #arch=modelq10.to_json()
    #with open( 'activations.json', 'w') as arch_file:
    #    arch_file.write(arch)
    #print('summary',modelq10.summary())
    #{layer.name: layer.get_weights() for layer in modelq10.layers}

    # trainmeanPredict = modelmean.predict(trainX)
    # testmeanPredict = modelmean.predict(testX)
    # trainstdPredict = modelstd.predict(trainX)
    # teststdPredict = modelstd.predict(testX)
    # print('trainmeanPredict',trainmeanPredict)
    i = 0
    j = look_back
    actual_quantile_interval = []
    steps = tw
    positive = True
    anomalies=[]
    finalres_q10 = []
    finalres_q90 = []
    dataframe = read_csv('/content/aws5f5533_data.csv', usecols=[1], engine='python')
    dataset = dataframe.values
    dataset = scaler.fit_transform(dataset)
    ts = dataset
    ts_accumulate=[]
    comparison_dataset=[]
    while j <= len(dataset):
        q50_array = []


        temp = dataset[i:j]
        actual_quantile_interval.append(
            numpy.absolute(numpy.quantile(dataset[i + 1:j + 1], lower_q_threshold) - numpy.quantile(dataset[i + 1:j + 1], upper_q_threshold)))
        print('print here', temp)

        for m in range(0, len(temp), steps):
            q50array = []
            q50 = numpy.quantile(temp[m:m + steps], upper_q_threshold)
            q50array.append(q50)
            q50_array.append(q50array)

        # print('stdarray1', std_array)
        # std_array = numpy.array(std_array)
        # print('stdarray2', std_array)
        # std_array = numpy.reshape(std_array, (std_array.shape[0],std_array.shape[1], 1))
        # print('stdarray3', std_array)
        # print('avg_array',avg_array)
        # avg_array = numpy.array(avg_array)
        # avg_array = numpy.reshape(avg_array, (avg_array.shape[0],1, 1))
        final_q50_array = []
        final_q50_array.append(q50_array)
        print('final_q10_array', final_q50_array)
        q50_predict = modelq10.predict(final_q50_array)
        print('q50_predict', q50_predict)

        if j+1 < len(dataset) :

            diff=q50_predict-dataset[j+1]
            print('data',dataset[j+1],'diff',diff)
            anomalies.append(diff)
            comparison_dataset.append(dataset[j+1])
            #dataset=numpy.delete(dataset,j+1)
            #print('length',len(dataset))
        #finalres_q10.append(q10_predict)
        #finalres_q90.append(q90_predict)

        j = j + 1
        i = i + 1

    # print('finalres',finalres)
    '''prediction_array_q10 = []
    prediction_array_q90 = []'''
    anomalies_array=[]

    for h in range(len(anomalies)):
        internal = anomalies[h]
        internal_array = []
        #internal_array.append(internal[0])
        anomalies_array.append(internal[0])
    anomalies_array = scaler.inverse_transform(anomalies_array)
    comparison_dataset=scaler.inverse_transform(comparison_dataset)
    print(anomalies_array)
    for itr in range(len(anomalies_array)):
        print('data',comparison_dataset[itr],'diff',anomalies_array[itr])
    '''print('anomaly length',len(anomalies_array))'''
    '''ts_accumulate_another=[]
    for h in range(len(finalres_q10)):
        internal = finalres_q10[h]
        internal_q90 = finalres_q90[h]
        prediction_array_q10.append(internal[0])
        prediction_array_q90.append(internal_q90[0])
    for g in range(len(ts_accumulate)):
        internal=[]
        internal.append(ts_accumulate[g])
        #internal_q90 = finalres_q90[h]
        ts_accumulate_another.append(internal)
    finalres_q10 = scaler.inverse_transform(prediction_array_q10)
    finalres_q90 = scaler.inverse_transform(prediction_array_q90)
    #print('finalres', finalres_q10)'''
    '''trunc_finalres = []
    for g in range(len(finalres)):
        trunc_finalres.append(finalres[g])'''
    '''ts = ts[look_back:]
    ts = scaler.inverse_transform(ts)
    ts_accumulate=scaler.inverse_transform(ts_accumulate_another)
    print('lenght', len(ts_accumulate), 'actual_quantile_interval', len(finalres_q10))'''
    '''ts_array = []
    for g in range(len(ts)):
        ts_array.append(ts[g])'''
    '''finalres_q10_array=[]
    finalres_q90_array=[]
    for g in range(len(finalres_q10)-1):
        finalres_q10_array.append(finalres_q10[g])
        finalres_q90_array.append(finalres_q90[g])
    prediction_errors = []'''
   # for y in range(len(finalres_q10_array)):
    #    print('ts_accumulate',ts_accumulate[y], 'finalres_q10_array', finalres_q10_array[y], 'finalres_q90_array', finalres_q90_array[y])
        #prediction_errors.append(numpy.absolute(actual_quantile_interval[y] - trunc_finalres[y]))
    # testScore = math.sqrt(mean_squared_error(ts, trunc_finalres))
    # print('Test Score: %.2f RMSE' % (testScore))
    anomalies = identify_anomaly(anomalies_array)
    print('anomaly_iden', anomalies)
    print('anomaly_iden size', len(anomalies))
    print(intersection(anomalies, anomaly_res))
    plt.figure(figsize=(12, 6))
    #plt.plot(ts, label="passengers")
    plt.plot(comparison_dataset, label="dataset")
    plt.plot(anomalies_array, label="difference")
    #plt.plot(newdata['ewa'], label="ewa")
    plt.legend(loc='best')

    plt.show()

    plt.figure(figsize=(12, 6))
    # plt.plot(ts, label="passengers")
    plt.plot(first_layer, label="first_layer")
    plt.plot(second_layer, label="second_layer")
    plt.plot(third_layer, label="third_layer")
    plt.plot(fourth_layer, label="fourth_layer")
    plt.plot(swish_layer,label='pef')
    # plt.plot(newdata['ewa'], label="ewa")
    plt.legend(loc='best')

    plt.show()
    print('alpha_values',alpha_values)
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score
RANDOM_SEED = 2021
TEST_PCT = 0.3
if __name__ == '__main__':

    LABELS = ["Normal","Fraud"]
    dataframe = pd.read_csv('/content/speed_t4013.csv',usecols=[1], engine='python')
    dataset = dataframe.values
    print(dataset)
    # dataframe=dataframe.diff(axis = 0, periods = 1)
    # dataset = dataframe.dropna().values
    dataset = dataset.astype('float32')
    # normalize the dataset
    print('dataset', dataset)

    sc = StandardScaler()
    result = sc.fit_transform(dataset.reshape(-1, 1))
    train_size = int(len(dataset) * 0.7)
    test_size = len(dataset) - train_size
    train_data1, test_data1 = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    train_data, test_data = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    min_val = tf.reduce_min(train_data)
    max_val = tf.reduce_max(train_data)
    train_data = (train_data - min_val) / (max_val - min_val)
    test_data = (test_data - min_val) / (max_val - min_val)
    train_data = tf.cast(train_data, tf.float32)
    test_data = tf.cast(test_data, tf.float32)
    nb_epoch = 50
    batch_size = 64
    input_dim = train_data.shape[1]  # num of columns, 30
    encoding_dim = 14
    hidden_dim_1 = int(encoding_dim / 2)  #
    hidden_dim_2 = 4
    learning_rate = 1e-7
    input_layer = tf.keras.layers.Input(shape=(input_dim,))
    encoder = tf.keras.layers.Dense(encoding_dim, activation="tanh",
                                    activity_regularizer=tf.keras.regularizers.l2(learning_rate))(input_layer)
    encoder = tf.keras.layers.Dropout(0.2)(encoder)
    encoder = tf.keras.layers.Dense(hidden_dim_1, activation='relu')(encoder)
    encoder = tf.keras.layers.Dense(hidden_dim_2, activation=tf.nn.leaky_relu)(encoder)
    # Decoder
    decoder = tf.keras.layers.Dense(hidden_dim_1, activation='relu')(encoder)
    decoder = tf.keras.layers.Dropout(0.2)(decoder)
    decoder = tf.keras.layers.Dense(encoding_dim, activation='relu')(decoder)
    decoder = tf.keras.layers.Dense(input_dim, activation='tanh')(decoder)
    autoencoder = tf.keras.Model(inputs=input_layer, outputs=decoder)
    autoencoder.summary()
    cp = tf.keras.callbacks.ModelCheckpoint(filepath="autoencoder_fraud.h5",
                                            mode='min', monitor='val_loss', verbose=2, save_best_only=True)
    # define our early stopping
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.0001,
        patience=10,
        verbose=1,
        mode='min',
        restore_best_weights=True)
    autoencoder.compile(metrics=['accuracy'],
                        loss='mean_squared_error',
                        optimizer='adam')
    history = autoencoder.fit(train_data, train_data,
                              epochs=nb_epoch,
                              batch_size=batch_size,
                              shuffle=True,
                              validation_data=(test_data, test_data),
                              verbose=1,
                              callbacks=[cp, early_stop]
                              ).history
    plt.plot(history['loss'], linewidth=2, label='Train')
    plt.plot(history['val_loss'], linewidth=2, label='Test')
    plt.legend(loc='upper right')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    # plt.ylim(ymin=0.70,ymax=1)
    plt.show()
    test_x_predictions = autoencoder.predict(test_data)
    mse = np.mean(np.power(test_data - test_x_predictions, 2), axis=1)
    print('mse',mse)
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    # plt.plot(ts, label="passengers")
    ax1.set_title('comparison')
    ax1.set_xlabel('time')
    ax1.set_ylabel('mse')
    ax1.plot(mse, label="dataset")

    ax2.set_xlabel('time')
    ax2.set_ylabel('value')
    ax2.plot(test_data1, label="test_data")
    # plt.plot(anomalies_array, label="difference")
    # plt.plot(newdata['ewa'], label="ewa")
    plt.legend(loc='best')
    plt.show()


    cutoff = (2*np.std(mse)+np.mean(mse))
    cutoff_1 = (-2*np.std(mse)+np.mean(mse))                    # decide on a cutoff limit
    prediction = np.zeros_like(mse)    # initialise a matrix full with zeros
    prediction[mse > cutoff] = 1
    prediction[mse < cutoff_1] = 1       
    #print(prediction)
    cutoff2 = ((2*np.std(test_data))+np.mean(test_data))
    cutoff3 = ((-2*np.std(test_data))+np.mean(test_data))
    test = np.zeros_like(test_data)
    test[test_data > cutoff2] = 1
    test[test_data < cutoff3] = 1
    #print(test)

    confusion_matrix(test,prediction)
    precision_score(test,prediction)
    recall_score(test,prediction)
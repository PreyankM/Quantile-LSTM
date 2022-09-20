import os
import sys
import time
import logging
import importlib

#Import data manipulation libraries
import numpy as np
import pandas as pd
import collections
from tqdm import tqdm

#Import visualization libraries
import matplotlib.pyplot as plt

#Importing ML/DL libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc,precision_recall_fscore_support, average_precision_score
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix,accuracy_score

from tensorflow.keras  import initializers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout
#from tensorflow.keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.layers import Input, BatchNormalization, LeakyReLU, Dense, Reshape, Flatten, Activation
from tensorflow.keras.layers import Dropout, multiply, GaussianNoise, MaxPooling2D, concatenate
import pickle
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import random
random.seed(123)

#data_path = './data/'
#loading the pickled file

#filename = data_path + 'preprocessed_data_full.pkl'
#input_file = open(filename,'rb')
#preprocessed_data = pickle.load(input_file)
#input_file.close()
#for key in preprocessed_data:
  #  print(key)


def get_generator(optimizer):
    generator = Sequential()
    generator.add(Dense(64, input_dim=1, kernel_initializer=initializers.glorot_normal(seed=42)))
    generator.add(Activation('tanh'))

    generator.add(Dense(128))
    generator.add(Activation('tanh'))

    generator.add(Dense(256))
    generator.add(Activation('tanh'))

    generator.add(Dense(256))
    generator.add(Activation('tanh'))

    generator.add(Dense(512))
    generator.add(Activation('tanh'))

    generator.add(Dense(1, activation='tanh'))

    generator.compile(loss='binary_crossentropy', optimizer=optimizer,run_eagerly=True)

    return generator


def get_discriminator(optimizer):
    discriminator = Sequential()

    discriminator.add(Dense(256, input_dim=1, kernel_initializer=initializers.glorot_normal(seed=42)))
    discriminator.add(Activation('relu'))
    discriminator.add(Dropout(0.2))

    discriminator.add(Dense(128))
    discriminator.add(Activation('relu'))
    discriminator.add(Dropout(0.2))

    discriminator.add(Dense(128))
    discriminator.add(Activation('relu'))
    discriminator.add(Dropout(0.2))

    discriminator.add(Dense(128))
    discriminator.add(Activation('relu'))
    discriminator.add(Dropout(0.2))

    discriminator.add(Dense(128))
    discriminator.add(Activation('relu'))
    discriminator.add(Dropout(0.2))

    discriminator.add(Dense(1))
    discriminator.add(Activation('sigmoid'))

    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer,run_eagerly=True)

    return discriminator


def get_gan_network(discriminator, generator, optimizer, input_dim=1):
    discriminator.trainable = False
    gan_input = Input(shape=(input_dim,))
    x = generator(gan_input)
    gan_output = discriminator(x)

    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=optimizer,run_eagerly=True)

    return gan

if __name__ == '__main__':
    print('hi')
    learning_rate = 0.00001
    batch_size = 512
    epochs = 10
    adam = Adam(lr=learning_rate, beta_1=0.5)
    dataframe = pd.read_csv('/content/speed_t4013_train.csv', usecols=[2], engine='python')
    df = pd.read_csv('/content/speed_t4013_labelled.csv', usecols=[2], engine='python')
    dataset = dataframe.values
    df=df.values
    print(dataset)
    # dataframe=dataframe.diff(axis = 0, periods = 1)
    # dataset = dataframe.dropna().values
    dataset = dataset.astype('float32')
    df=df.astype('float32')
    # normalize the dataset
    #print('dataset', dataset)
  

    sc = StandardScaler()
    result = sc.fit_transform(dataset.reshape(-1, 1))
    train_size = int(len(dataset))
    test_size = int(len(df))
    train_data1, test_data1 = dataset, df
    train_data, test_data = dataset, df
    min_val = tf.reduce_min(train_data)
    max_val = tf.reduce_max(train_data)
    train_data = (train_data - min_val) / (max_val - min_val)
    test_data = (test_data - min_val) / (max_val - min_val)
    train_data = tf.cast(train_data, tf.float32)
    test_data = tf.cast(test_data, tf.float32)

    # Calculating the number of batches based on the batch size
    batch_count = train_data.shape[0] // batch_size
    pbar = tqdm(total=epochs * batch_count)
    gan_loss = []
    discriminator_loss = []

    # Inititalizing the network
    generator = get_generator(adam)
    discriminator = get_discriminator(adam)
    gan = get_gan_network(discriminator, generator, adam, input_dim=1)

    for epoch in range(epochs):
        for index in range(batch_count):
            pbar.update(1)
            # Creating a random set of input noise and images
            noise = np.random.normal(0, 1, size=[batch_size, 1])

            # Generate fake samples
            generated_images = generator.predict_on_batch(noise)

            # Obtain a batch of normal network packets
            image_batch = train_data[index * batch_size: (index + 1) * batch_size]

            X = np.vstack((generated_images, image_batch))
            y_dis = np.ones(2 * batch_size)
            y_dis[:batch_size] = 0

            # Train discriminator
            discriminator.trainable = True
            d_loss = discriminator.train_on_batch(X, y_dis)

            # Train generator
            noise = np.random.uniform(0, 1, size=[batch_size, 1])
            y_gen = np.ones(batch_size)
            discriminator.trainable = False
            g_loss = gan.train_on_batch(noise, y_gen)

            # Record the losses
            discriminator_loss.append(d_loss)
            gan_loss.append(g_loss)

        print("Epoch %d Batch %d/%d [D loss: %f] [G loss:%f]" % (epoch, index, batch_count, d_loss, g_loss))
        test_x_predictions = discriminator.predict(test_data)
        #print('test_x_predictions',test_x_predictions)
        #per = np.percentile(test_x_predictions, 1)

        y_pred = test_x_predictions.copy()
        y_pred = np.array(y_pred)

        perup = np.mean(test_x_predictions) + 2*np.std(test_x_predictions)
        perdown = np.mean(test_x_predictions) - 2*np.std(test_x_predictions)

        # Thresholding based on the score
        for i in range(len(y_pred)):
          if(test_x_predictions[i] >= perup or test_x_predictions[i] <= perdown):
            y_pred[i] = 1
          else:
            y_pred[i] = 0
          
        #inds = (y_pred < perup and y_pred > perdown)
        #inds_comp = (y_pred >= perup and y_pred <= perdown)

        #y_pred[inds] = 0
        #y_pred[inds_comp] = 1
        print(y_pred)
        for itr in range(len(y_pred)):
            print('data',df[itr],'pred',y_pred[itr])


    err = [15,11]


    #making the test data binary using know anamolies
    prediction = np.zeros_like(test_data)
    for i in range(len(test_data)):
        for j in err:
            check = (j - min_val) / (max_val - min_val)
            if(test_data[i] == check):
                prediction[i] = 1
                print(test_data[i])

    #print((y_pred))

    precision, recall, f1, _ = precision_recall_fscore_support(prediction, y_pred, average='binary')
    print('Accuracy Score :', accuracy_score(prediction, y_pred))
    print('Precision :', precision)
    print('Recall :', recall)
    print('F1 :', f1)
    confusion_matrix(prediction,y_pred)        
# Credit fraud dataset is from https://www.kaggle.com/mlg-ulb/creditcardfraud,
# you could download it as you wish

import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    data = pd.read_csv('/content/final.csv')
    data.drop(['Date'], axis=1, inplace=True)
    data.rename(columns={'ana':'class'}, inplace=True)

    data.to_csv('/content/test.csv', index=False)
    train, test = train_test_split(data, test_size=0.2)

    train.to_csv('/content/train.csv', index=False)
    test.to_csv('/content/test.csv', index=False)

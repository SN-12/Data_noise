
# import libraries
import pandas as pd
import requests
from io import StringIO

"""## 1.1 - Downloading data for Terminal 500"""

orig_url='https://drive.google.com/file/d/1l5Jxt-E1qw-MCs9ZrwT9NK9pFvu-xQyb/view'

file_id = orig_url.split('/')[-2]
dwn_url='https://drive.google.com/uc?export=download&id=' + file_id
url = requests.get(dwn_url).text
csv_raw = StringIO(url)
df_500 = pd.read_csv(csv_raw)
print(df_500.head())

"""## 1.2 - Downloading data for Terminal 515"""

orig_url='https://drive.google.com/file/d/1RcfD5nZe59esG9XfSttuhwkcp99bxac1/view'

file_id = orig_url.split('/')[-2]
dwn_url='https://drive.google.com/uc?export=download&id=' + file_id
url = requests.get(dwn_url).text
csv_raw = StringIO(url)
df_515 = pd.read_csv(csv_raw)
print(df_515.head())

"""## 1.3 - Downloading data for Terminal 521"""

orig_url='https://drive.google.com/file/d/1-l23Hgx9fBwg5-EGPpnu-YCnJZ9hXHkX/view'

file_id = orig_url.split('/')[-2]
dwn_url='https://drive.google.com/uc?export=download&id=' + file_id
url = requests.get(dwn_url).text
csv_raw = StringIO(url)
df_521 = pd.read_csv(csv_raw)
print(df_521.head())

"""## 1.4 - Prepare Data with Pandas

### 1.4.1 - Use noise levels ranging from May 2019 to end of January 2020
"""

start_date = '2019-05-01 00:00:00'
end_date = '2020-01-31 23:59:59'

mask = (df_515['measured_time'] >= start_date) & (df_515['measured_time'] <= end_date)
df = df_515.loc[mask]
df

"""### 1.4.2 - Extract date and hour from datetime

"""

df.measured_time = pd.to_datetime(df.measured_time)
df['date'] = df.measured_time.dt.date
df['hour'] = df.measured_time.dt.hour
df

"""### 1.4.3 - Group by date and hour"""

df1 = df.groupby(['date','hour'], as_index=False).sum()
df1

"""### 1.4.4 - Merge data and hour into one additional column and ensure datetime type"""

# Necessary because we lose the measured_time column when we group and sum
df1['date'] = pd.to_datetime(df1.date, cache=True) 
df1['hour'] = pd.to_timedelta(df1.hour, unit='h')
df1['datetime'] = df1['date'] + df1['hour']
df1

"""### 1.4.5 - Calculating Average Noise Value per Hour"""

'''
This function calculates the Average Noise Value per hour
Returns a series
'''
def calculateAverageNoise(row):
  numerator = 0
  denominator = 0
  for i in range(77):
    name = 'dataNoise'+str(i)
    numerator += (i+35) * row[name] 
    denominator += row[name]
  avgNoise = numerator / denominator
  return pd.Series({'average_noise':avgNoise})

# Join average_noise series to dataframe
df2 = df1.join(df1.apply(calculateAverageNoise, axis=1))

df2

"""### 1.4.6 - Drop rows where `average_noise`is NaN, if exists

"""

df2 = df2.dropna(subset=['average_noise'])
len(df2)

"""### 1.4.7 - Keep only `datetime`and `average_noise` columns"""

df2 = df2.loc[:,['datetime','average_noise']]
df2.head(5)

"""### 1.4.8 - Sort rows according to datetime"""

df2.sort_values('datetime', inplace=True, ascending=True)
df2

"""# 2 - Plotting to show Data Evolution Over Time

## 2.1 - Use a time range of two weeks going from May 1st 2019 until mid May
"""

start_date = '2019-05-01 00:00:00'
end_date = '2019-05-15 00:00:00'
mask = (df2['datetime'] >= start_date) & (df2['datetime'] < end_date)
df3 = df2.loc[mask]
df3

"""## 2.2 - Plot average_noise for all the time range (515 only)

"""

import matplotlib.pyplot as plt

df3.set_index('datetime',inplace=True)
df3.plot(figsize=(15,5))
plt.ylabel('Average Noise (dB)')
plt.show()

"""# 3 - Build dataset from `average_noise` data

## 3.1 - Build dataset as a numpy array from column `average_noise` of the Pandas dataframe
"""

import numpy as np

dataset = df2['average_noise'].values
print(dataset.shape)
dataset = np.reshape(dataset, (-1,1))
print(dataset.shape)

"""## 3.2 - Normalize the data by mapping them in $[0;1]$. """

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

"""## 3.3 - Split the dataset into training data and testing data, with 90% of the samples for training"""

train_size = int(len(dataset) * 0.90)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

print(train.shape)
print(test.shape)

"""## 3.4 - Use TimeseriesGenerator to build train and test sets to be fed in Keras model"""

from keras.preprocessing.sequence import TimeseriesGenerator

lookback = 12
train_data_gen = TimeseriesGenerator(train,train,length=lookback,sampling_rate=1,stride=1,batch_size=10)
print(len(train_data_gen))

test_data_gen = TimeseriesGenerator(test,test,length=lookback,sampling_rate=1,stride=1,batch_size=1)
print(len(test_data_gen))

"""# 4 - Studied Machine Learning Models

## 4.1 - Stacked Long Short-Term Memory (Stacked-LSTM)

### 4.1.1 - Building Model
"""

from keras.models import Sequential
from keras.layers import Dense,Dropout,LSTM
from keras.callbacks import EarlyStopping

model = Sequential()
model.add(LSTM(150, activation='relu', input_shape=(lookback,1),return_sequences=True))
model.add(LSTM(100, activation='relu'))
model.add(Dense(1, activation='linear'))

model.summary()

model.compile(loss='mean_squared_error', optimizer='rmsprop')

from google.colab import drive
drive.mount("/content/drive")

import tensorflow as tf

# Save best model
checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath="/content/drive/MyDrive/StackedLSTM.hdf5", monitor='val_loss', verbose=1, save_best_only=True)
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(train_data_gen, epochs=25, validation_data=test_data_gen, 
                    callbacks=[earlystopping, checkpointer], verbose=1,shuffle=True)

# Load the saved model
import keras
model = keras.models.load_model('/content/drive/MyDrive/StackedLSTM.hdf5')

"""### 4.1.2 - Prediction"""

# Predicted train and test values
train_predict = model.predict(train_data_gen)
test_predict = model.predict(test_data_gen)
print("Train and Test predictions shape")
print(train_predict.shape)
print(test_predict.shape)

def get_y_from_generator(gen):
    '''
    Get all targets y from a TimeseriesGenerator instance.
    '''
    y = None
    for i in range(len(gen)):
        batch_y = gen[i][1]
        if y is None:
            y = batch_y
        else:
            y = np.append(y, batch_y)
    y = y.reshape((-1,1))
    print(y.shape)
    return y

Y_train = get_y_from_generator(train_data_gen)
Y_test = get_y_from_generator(test_data_gen)
# Mapping back predicted train values
train_predict = scaler.inverse_transform(train_predict)
Y_train = scaler.inverse_transform(Y_train)
# Mapping back predicted test values
test_predict = scaler.inverse_transform(test_predict)
Y_test = scaler.inverse_transform(Y_test)
print("Expected Train and Test shape")
print(Y_train.shape)
print(Y_test.shape)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

print(Y_train[0])
print(train_predict[:,0])
print('Train Root Mean Squared Error:',np.sqrt(mean_squared_error(Y_train, train_predict)))
print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(Y_test, test_predict)))

plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')
plt.show();

import seaborn as sns
sns.set_context("paper", font_scale=1.3)
sns.set_style('white')

X_data = [x for x in range(200)]
plt.figure(figsize=(8,4))
plt.plot(X_data, Y_test[:200], marker='.', label="actual")
plt.plot(X_data, test_predict[:200], 'r', label="prediction")
plt.tight_layout()
sns.despine(top=True)
plt.subplots_adjust(left=0.07)
plt.ylabel('Average Noise Level', size=15)
plt.xlabel('Time step', size=15)
plt.legend(fontsize=15)
plt.show();

"""## 4.2 - Combination of Convolutional Neural Network and LSTM (CNN-LSTM)

### 4.2.1 - Building Model
"""

from keras.layers import Conv1D
from keras.initializers import GlorotUniform

model2 = Sequential()

#Part 1
model2.add(Conv1D(16, padding='same', kernel_size=3, kernel_initializer=GlorotUniform(), input_shape=(lookback,1)))
model2.add(Conv1D(16, padding='same', kernel_size=3, kernel_initializer=GlorotUniform()))
model2.add(Conv1D(16, padding='same', kernel_size=3, kernel_initializer=GlorotUniform()))

# Part 2
model2.add(LSTM(150, activation='relu', input_shape=(lookback,1),return_sequences=True))
model2.add(LSTM(100, activation='relu'))
model2.add(Dense(1, activation='linear'))

model2.summary()

model2.compile(loss='mean_squared_error', optimizer='rmsprop')

checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath="/content/drive/MyDrive/CNN-LSTM.hdf5", monitor='val_loss', verbose=1, save_best_only=True)
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model2.fit(train_data_gen, epochs=100, validation_data=test_data_gen, 
                    callbacks=[earlystopping, checkpointer], verbose=1,shuffle=True)

"""### 4.2.2 - Prediction"""

# Predicted train and test values
train_predict = model2.predict(train_data_gen)
test_predict = model2.predict(test_data_gen)
print("Train and Test predictions shape")
print(train_predict.shape)
print(test_predict.shape)

def get_y_from_generator(gen):
    '''
    Get all targets y from a TimeseriesGenerator instance.
    '''
    y = None
    for i in range(len(gen)):
        batch_y = gen[i][1]
        if y is None:
            y = batch_y
        else:
            y = np.append(y, batch_y)
    y = y.reshape((-1,1))
    print(y.shape)
    return y

Y_train = get_y_from_generator(train_data_gen)
Y_test = get_y_from_generator(test_data_gen)
# Mapping back predicted train values
train_predict = scaler.inverse_transform(train_predict)
Y_train = scaler.inverse_transform(Y_train)
# Mapping back predicted test values
test_predict = scaler.inverse_transform(test_predict)
Y_test = scaler.inverse_transform(Y_test)
print("Expected Train and Test shape")
print(Y_train.shape)
print(Y_test.shape)

from sklearn.metrics import mean_squared_error

print(Y_train[0])
print(train_predict[:,0])
# print('Train Mean Absolute Error:', mean_absolute_error(Y_train, train_predict))
print('Train Root Mean Squared Error:',np.sqrt(mean_squared_error(Y_train, train_predict)))
# print('Test Mean Absolute Error:', mean_absolute_error(Y_test, test_predict))
print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(Y_test, test_predict)))

plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')
plt.show();

import seaborn as sns
sns.set_context("paper", font_scale=1.3)
sns.set_style('white')

X_data = [x for x in range(200)]
plt.figure(figsize=(8,4))
plt.plot(X_data, Y_test[:200], marker='.', label="actual")
plt.plot(X_data, test_predict[:200], 'r', label="prediction")
plt.tight_layout()
sns.despine(top=True)
plt.subplots_adjust(left=0.07)
plt.ylabel('Average Noise Level', size=15)
plt.xlabel('Time step', size=15)
plt.legend(fontsize=15)
plt.show();

"""## 4.3 - Gradient Boosting (LightGBM)

### 4.3.1 - Building Model
"""

import lightgbm as lgb

df2

y = df2['average_noise'].values
X = df2['datetime'].values

X = X.reshape((-1,1))

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.10)

lgb_train = lgb.Dataset(x_train, y_train)
lgb_valid = lgb.Dataset(x_test, y_test, reference=lgb_train)

params = {
    'task':'train',
    'boosting_type':'gbdt',
    'objective':'regression',
    'metric':{'rmse'},
    'num_leaves':11,
    'num_iterations':1000,
    'feature_fraction':0.9,
    'bagging_fraction':0.8,
    'bagging_freq':5,
    'learning_rate':0.05,
    'N_estimator':500,
    'verbose':1
}

model = lgb.train(params, lgb_train, num_boost_round=20, valid_sets=[lgb_train, lgb_valid],early_stopping_rounds=10)

y_test = model.predict(x_test)
y_test

"""### 4.3.2 - Prediction"""

import seaborn as sns
sns.set_context("paper", font_scale=1.3)
sns.set_style('white')

X_data = [x for x in range(200)]
plt.figure(figsize=(8,4))
plt.plot(X_data, Y_test[:200], marker='.', label="actual")
plt.plot(X_data, y_test[:200], 'r', label="prediction")
plt.tight_layout()
sns.despine(top=True)
plt.subplots_adjust(left=0.07)
plt.ylabel('Average Noise Level', size=15)
plt.xlabel('Time step', size=15)
plt.legend(fontsize=15)
plt.show();

"""# 5 - Detection of False Data Injection Attacks

## 5.1 - Make copy of original dataset
"""

Y_test_injected = Y_test.copy()

Y_test_injected.shape

"""## 5.2 - Initialize list of Attack Locations"""

Attacked = [0]*len(Y_test)
len(Attacked)

"""## 5.3 - Randomly disturb the input data by adding / subtracting values to some of the original average noise level values"""

for i in range(30):
  position = randint(0,645)
  if (i%2==0):
    Y_test_injected[position] += randint(10,15)
  else:
    Y_test_injected[position] -= randint(10,15)
  Attacked[position] = 1

"""## 5.4 - Plot Original Data, Injected Data, and Attack Locations"""

sns.set_context("paper", font_scale=1.3)
sns.set_style('white')

X_data = [x for x in range(len(Y_test))]
plt.figure(figsize=(15,4))
plt.plot(X_data, Y_test[:], label="original data")
plt.plot(X_data, Y_test_injected[:], color='orange', label="data obtained after false data injections")
plt.plot(X_data, Attacked[:], color='green', label='location of data attack')
plt.tight_layout()
sns.despine(top=True)
plt.subplots_adjust(left=0.07)
plt.ylabel('Average Noise Level', size=15)
plt.xlabel('Time step', size=15)
plt.legend(fontsize=15)
plt.show();

"""## 5.5 - Initialize list of Anomalies Locations"""

Anomalies = [0]*len(Y_test)
len(Anomalies)

"""## 5.6 - Find Anomalies Using Threshold of 5dB"""

for i in range(len(Y_test)):
  # if valu
  if (abs(Y_test_injected[i] - test_predict[i]) > 5):
    Anomalies[i] = 1

"""## 5.7 - Plot Injected Data, Predicted Data, and Anomaly Locations Detected"""

sns.set_context("paper", font_scale=1.3)
sns.set_style('white')

X_data = [x for x in range(len(Y_test))]
plt.figure(figsize=(15,4))
plt.plot(X_data, Y_test_injected[:], label="real data with false data injections")
plt.plot(X_data, test_predict[:], color='orange', label="predicted")
plt.plot(X_data, Anomalies[:], color='red', label='anomaly detected')
plt.tight_layout()
sns.despine(top=True)
plt.subplots_adjust(left=0.07)
plt.ylabel('Average Noise Level', size=15)
plt.xlabel('Time step', size=15)
plt.legend(fontsize=15)
plt.show();

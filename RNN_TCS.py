
import numpy as np   #contain maths
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd #to import dataset and to manage data set

def prediction():
    ##### Data Preprocessing#####

    #Importing the training set#
    dataset_train=pd.read_csv('TCS_Train.csv')
    training_set=dataset_train.iloc[:, 1:2].values

    #Feature Scaling#    #Standardization OR Normalisation
    from sklearn.preprocessing import  MinMaxScaler
    sc=MinMaxScaler(feature_range=(0, 1)) #to make all stock prices between o and 1
    training_set_scaled=sc.fit_transform(training_set)


    #Creating a data structure with 60 timesteps and 1 output#
    #so RNN will try to learn 60 previuos observation at time t and try to predict 1 output#
    X_train=[]
    y_train=[]
    for i in range(60, 2460):
        X_train.append(training_set_scaled[i-60:i, 0])
        y_train.append(training_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    #Reshaping# to add new dimention to numpy array
    X_train=np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


    ##### Building the RNN#####


    #Import Keras Libraries and packages#
    from keras.models import Sequential
    from keras.layers import  Dense
    from keras.layers import  LSTM
    from keras.layers import Dropout

    #Initialising the RNN#
    regressor=Sequential()

    #Adding First LSTM layer and some dropout regularisation#
    regressor.add(LSTM(units = 50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    regressor.add(Dropout(0.20))

    #Adding Second LSTM layer and some dropout regularisation#
    regressor.add(LSTM(units = 50, return_sequences=True))
    regressor.add(Dropout(0.20))

    #Adding Third LSTM layer and some dropout regularisation#
    regressor.add(LSTM(units = 50, return_sequences=True))
    regressor.add(Dropout(0.20))

    #Adding Fourth LSTM layer and some dropout regularisation#
    regressor.add(LSTM(units = 50)) #As its last LSTM layer so we are not going to pass any sequence hense return_sequences=FALSE OR remove the parameter
    regressor.add(Dropout(0.20))

    #Adding the output Layer#
    regressor.add(Dense(units=1))

    #Compiling the RNN#
    regressor.compile(optimizer='adam', loss='mean_squared_error')
    # for regression loss is measured by mean_squared_error#

    #Fitting the RNN to training set#
    regressor.fit(X_train, y_train, batch_size=32, epochs=100)




    #####Making the Predictions and Visualising the results#####

    #Getting the real stock price of 2017#
    dataset_test = pd.read_csv('TCS_Test.csv')
    real_Stock_price = dataset_test.iloc[:, 1:2].values

    #Getting the predicted stock price of 2017#
    dataset_total = pd.concat((dataset_train['Close'], dataset_test['Close']), axis=0)
    inputs = dataset_total[len(dataset_total)-len(dataset_test) - 60:].values
    inputs = inputs.reshape(-1, 1)
    inputs = sc.transform(inputs)
    X_test = []

    for i in range(60, 81):
        X_test.append(inputs[i-60:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_Stock_price = regressor.predict(X_test)
    predicted_Stock_price = sc.inverse_transform(predicted_Stock_price)

    import math
    from sklearn.metrics import mean_squared_error
    rmse = math.sqrt(mean_squared_error(real_Stock_price, predicted_Stock_price))
    print("The Root Mean Squred Error Is:- ", rmse)

    #Visualising the results#
    plt.plot(real_Stock_price, color='red', label = 'Real TCS Stock OPEN Price(Sep2020)')
    plt.plot(predicted_Stock_price, color='blue', label = 'Predicted TCS Stock OPEN Price(Sep2020)')
    plt.title('TCS Stock OPEN Price Prediction')
    plt.xlabel('Financial Days(September-2020)')
    plt.ylabel('TCS Stock OPEN Price')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    prediction()
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Description = This program uses an artificial recurrent neural network called LTSM (Long Short Term Memory) to 
#              predict the closing stock price of a corporation (Apple.in) using the past 60 days stock price.
#              LSTM are used for sequence prediction models.


# In[6]:


import math
import pandas as pd
import numpy as np
import pandas_datareader as web
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM 
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


# In[55]:


#Get the stock quote 

df = web.DataReader('MSFT' , data_source = 'yahoo' , start = '2012-01-01', end = '2020-03-31')

#show the data 
df


# In[57]:


# Get teh number of rows and columns 
df.shape


# In[58]:


#Visualise the closing price 

plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'])
#plt.plot(df['Close'])
plt.xlabel('Date' , fontsize = 18)
plt.ylabel('Close Price USD ($)' , fontsize = 18)
plt.show()


# In[59]:


# Create a new dataframe with only the close column 

data = df.filter(['Close'])

# convert the dataframe to a numpy array 

dataset = data.values

#Get the number of rows to train the model on 

training_data_len = math.ceil(len(dataset)*.8) 

print(training_data_len)


# In[60]:


#Scale the data 

scaler = MinMaxScaler(feature_range=(0,1)) # all the values will be between 0 and 1 
scaled_data = scaler.fit_transform(dataset)

print(scaled_data)


# In[61]:


#Create the training data set 
#Create the scaled training data set 
 
train_data = scaled_data[0:training_data_len, : ] # the third argument is a colon because we would want all the columns 
# Contains all the values from 0 to training_data_len.

# Split the data into x_train and y_train data sets 

x_train = [] #independent training variables

y_train = [] #dependent training variables  

for i in range( 60 , len(train_data)):
    x_train.append(train_data[i-60 :i , 0]) #data sets from i-60 to i, but not including i. 
    y_train.append(train_data[i , 0]) # only the 61st value, held at column 0 
    
    if i<=61:
        print(x_train)
        print(y_train)
        print()
 


# In[62]:


# Convert the x_train and the y_train to numpy arrays 

x_train, y_train = np.array(x_train), np.array(y_train)


# In[63]:


# Reshape the x_train data 

x_train = np.reshape(x_train , ( x_train.shape[0] , x_train.shape[1] , 1 )) # number of features is just 1, which is the closing price. 
x_train.shape 


# In[64]:


# Build the LSTM model 
model = Sequential()
model.add(LSTM( 50 , return_sequences = True , input_shape = ( x_train.shape[1] ,  1)))
model.add(LSTM( 50 , return_sequences = False)) # False in this case, because we are not going to be using any LSTM models for our model 
model.add(Dense(25))
model.add(Dense(1))


# In[65]:


# Compile the model 

model.compile(optimizer = 'adam' , loss = 'mean_squared_error') # adam is the name of an optimizer and 
# loss is another function for the complile function 
# an optimizer is to improve upon the loss function and the loss function is to measure how well the model did on training 


# In[66]:


#Train the model 
# fit is another word for Train in ML
# batch_size is the total number of training examples present in one training set 
#epoch is defined as the number of iterations when passed forward and backward through a neural network 
model.fit(x_train , y_train , batch_size = 1 , epochs = 1 ) 


# In[73]:


# Create a testing dataset 
# Create a new array containing scaled values frmo index 1600 to 2600
test_data = scaled_data[training_data_len - 60: , :]
# Create the data sets x_test and y_test 
x_test = []
y_test = dataset[training_data_len: , :]

# let us create the x_test set 
for i in range(60  , len(test_data)) :
    # here we are going to append the past 60 values to the x_test dataset
    x_test.append(test_data[ i - 60 : i , 0])
    


# In[74]:


# Convert the data to numpy array 
x_test = np.array(x_test)


# In[75]:


# Reshape the data 
# we are doing this because we would want our data to be 3 dimensional instead of 2 dimensional 
# this is also done because the LSTM model expects a 3 dimensional shape 

x_test = np.reshape(x_test , (x_test.shape[0] , x_test.shape[1] , 1 ))
# the third argument is 1 because we just want the output in 1 feature, which is the closing price : the 3rd argument 
# basically denotes how many features we would want to depict 
# the first argument denotes the number of samples or the number of rows  
# the second argument determines the number of time steps or the number of columns  


# In[76]:


# In this cell we would want the models predicted price values for the x_test dataset 
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions) # here we are basically unscaling the values 


# In[77]:


# Get root mean squared error (RMSE)
# it is a good measure of how accurate the model predicts the response 
# it is the standard deviation of the residuals 
# lower values of RMSE indicate a better fit 

rmse = np.sqrt(np.mean( predictions - y_test)**2) 

rmse

# a value of 0 for rmse is that the predictions were exact it basically means there was perfect prediction 
# for the RMSE, as compared with the testing data 


# In[79]:


#Plot the data 

train = data[0:training_data_len]
valid = data[training_data_len : ]
valid['Predictions'] = predictions 

#Visualize the model

plt.figure(figsize = (16 , 8) )
plt.title('Model')
plt.xlabel('Data' , fontsize = 18)
plt.ylabel('Close Price US ($)' , fontsize = 18)
plt.plot(train['Close'])
plt.plot(valid[['Close' , 'Predictions']])
plt.legend(['Train' , 'Valid' , 'Predictions'] , loc = 'top left')
plt.show()




# In[80]:


# Show the valid and the predicted prices 
valid


# In[83]:


# Get the quote 
microsoft_quote = web.DataReader('MSFT' , data_source = 'yahoo' , start = '2012-01-01' , end = '2020-03-31')
# Create a new dataframe 
new_df = microsoft_quote.filter(['Close'])
# Get the last 60 day Closing price values and convert the dataframe to an array 
last_60_days = new_df[-60 : ].values
# Scale the data for the values to be between 0 and 1 
last_60_days_scaled = scaler.transform(last_60_days)
# Create an empty list
X_test = []
#Append the past 60 days to 
X_test.append(last_60_days_scaled)
# Convert a X_test to numpy array 
X_test = np.array(X_test)
#Reshape the data to be 3d
X_test = np.reshape(X_test , (X_test.shape[0] , X_test.shape[1] , 1))
#Get the predicted scale price 
pred_price = model.predict(X_test)
#undo the scaling
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)


# In[86]:


# Get the quote 2 
microsoft_quote2 = web.DataReader('MSFT' , data_source = 'yahoo' , start = '2020-04-01' , end = '2020-04-01')
print(microsoft_quote2['Close'])


# In[ ]:





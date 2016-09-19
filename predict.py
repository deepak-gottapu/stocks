import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
stock = pd.read_csv('sphist.csv')
stock['Date'] = pd.to_datetime(stock['Date'])
stock.sort('Date', inplace = True)
print(stock.head())

def roll_mean(df,col,win):
    roll = pd.rolling_mean(df[col],window = win)
    roll = roll.shift(periods=1)
    return(roll)
    
def roll_std(df,col,win):
    roll = pd.rolling_std(df[col],window = win)
    roll = roll.shift(periods=1)
    return(roll)

cols = ['Close']
stock['rolling_close_mean_5'] = roll_mean(stock,cols,5)
stock['rolling_close_mean_365'] = roll_mean(stock,cols,365)
stock['rolling_close_std_5'] = roll_std(stock,cols,5)
stock['rolling_close_std_365'] = roll_std(stock,cols,365)
#stock['rolling_mean_ratio'] = roll_mean/roll_mean_365
#stock['rolling_std_ratio'] = roll_std_5/roll_std_365

cols = ['Volume']
stock['rolling_Volume_mean_5'] = roll_mean(stock,cols,5)
stock['rolling_Volume_mean_365'] = roll_mean(stock,cols,365)
stock['rolling_Volume_std_5'] = roll_std(stock,cols,5)
stock['rolling_Volume_std_365'] = roll_std(stock,cols,365)
#stock['rolling_Volume_mean_ratio'] = roll_mean/roll_mean_365
#stock['rolling_Volume_std_ratio'] = roll_std_5/roll_std_365

stock = stock[stock['Date']> datetime(year = 1951, month = 1, day = 2)]
stock.dropna(inplace=True,axis=0)

train = stock[stock['Date']< datetime(year = 2013, month = 1, day = 1)]
test = stock[stock['Date']>= datetime(year = 2013, month = 1, day = 1)]
#print(train.dtypes)

model = LinearRegression()
cols = ['rolling_close_mean_5','rolling_close_std_5']
#cols = ['rolling_close_mean_5','rolling_close_mean_365','rolling_close_std_5','rolling_close_std_365']
#cols = ['rolling_close_mean_5','rolling_close_mean_365','rolling_close_std_5','rolling_close_std_365','rolling_Volume_mean_5','rolling_Volume_mean_365','rolling_Volume_std_5','rolling_Volume_std_365'] 
model.fit(train[cols],train['Close'])
predictions_train = model.predict(train[cols])
predictions = model.predict(test[cols])
error_train = mean_absolute_error(train['Close'], predictions_train)
error = mean_absolute_error(test['Close'], predictions)
print(error_train, error)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv('/Users/kevinnguyen/Desktop/linear regression and eda/house_price_and_predictions/train.csv')
data_test = pd.read_csv('/Users/kevinnguyen/Desktop/linear regression and eda/house_price_and_predictions/test.csv')

#histogram to display SalePrice and model probability distribution
df['SalePrice'].describe().transpose()
sns.displot(df['SalePrice'], kde=True, bins=50)
plt.show()

print('Sale Price skewness:',df['SalePrice'].skew())

#heatmap displaying correlation between attributes
df.info()
df_num = df.select_dtypes(exclude='object')
df_num.shape
corr = df_num.corr()
corr_target = corr['SalePrice']
corr_target.sort_values(ascending=False)
corr_target = corr_target[corr_target>0.5]
corr_target.sort_values(ascending=False)
col_name = corr_target.keys()
corr_map = df_num[col_name].corr()

plt.figure(figsize=(7,7))
sns.heatmap(corr_map, square=True, annot=True)
plt.show()

#scatterplots for each attribute
col_name = ['OverallQual', 'OverallCond', 'GarageCars', 'FullBath', '1stFlrSF', 'GrLivArea', 'YearRemodAdd', 'SalePrice', 'YearBuilt', 'HalfBath']
df[col_name].isnull().sum()
new_data = df[col_name]

plt.figure(figsize=(15,10))
plt.subplot(3,3,1)
sns.scatterplot(x=new_data['OverallQual'], y=new_data['SalePrice'])
plt.subplot(3,3,2)
sns.scatterplot(x=new_data['OverallCond'], y=new_data['SalePrice'])
plt.subplot(3,3,3)
sns.scatterplot(x=new_data['GarageCars'], y=new_data['SalePrice'])
plt.subplot(3,3,4)
sns.scatterplot(x=new_data['FullBath'], y=new_data['SalePrice'])
plt.subplot(3,3,5)
sns.scatterplot(x=new_data['HalfBath'], y=new_data['SalePrice'])
plt.subplot(3,3,6)
sns.scatterplot(x=new_data['GrLivArea'], y=new_data['SalePrice'])
plt.subplot(3,3,7)
sns.scatterplot(x=new_data['YearRemodAdd'], y=new_data['SalePrice'])
plt.subplot(3,3,8)
sns.scatterplot(x=new_data['YearBuilt'], y=new_data['SalePrice'])
plt.subplot(3,3,9)
sns.scatterplot(x=new_data['1stFlrSF'], y=new_data['SalePrice'])
plt.show()

#training data set using SalePrice against GarageCars
X_train = new_data.drop(['SalePrice'], axis=1)
Y_train = new_data['SalePrice']
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.3, random_state=101)

lr = LinearRegression()
lr.fit(X_train,Y_train)

Y_predict = lr.predict(X_val)
np.sqrt(mean_squared_error(Y_val, Y_predict))
X_test = data_test[['OverallQual', 'OverallCond', 'GarageCars', 'FullBath', '1stFlrSF', 'GrLivArea', 'YearRemodAdd', 'YearBuilt', 'HalfBath']]
X_test.isnull().sum()
values = {'GarageCars':X_test['GarageCars'].median()}
X_test.fillna(values, inplace=True)
X_test.isnull().sum()
Y_test = lr.predict(X_test)

output = pd.DataFrame({'Id':data_test['Id'], 'SalePrice':Y_test})
output.to_csv('linreg.csv', index=False)
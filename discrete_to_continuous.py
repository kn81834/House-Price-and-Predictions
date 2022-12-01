import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.sparse import dok_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('/Users/kevinnguyen/Desktop/linear regression and eda/house_price_and_predictions/train.csv')
data_test = pd.read_csv('/Users/kevinnguyen/Desktop/linear regression and eda/house_price_and_predictions/test.csv')

#labelencode all categorical vairables to array
le = LabelEncoder()
dfle = df
dtle = data_test
obj_cols = df.select_dtypes(exclude=['int64','float64'])
col_names = []
for x in dfle:
    col_names.append(x)
#obj_cols.info()

'''
dfle.MSZoning = le.fit_transform(dfle.MSZoning)
dtle.MSZoning = le.fit_transform(dtle.MSZoning)
dfle.Street = le.fit_transform(dfle.Street)
dtle.Street = le.fit_transform(dtle.Street)
dfle.Alley = le.fit_transform(dfle.Alley)
dtle.Alley = le.fit_transform(dtle.Alley)
dfle.LotShape = le.fit_transform(dfle.LotShape)
dtle.LotShape = le.fit_transform(dtle.LotShape)
dfle.LandContour = le.fit_transform(dfle.LandContour)
dtle.LandContour = le.fit_transform(dtle.LandContour)
dfle.Utilities = le.fit_transform(dfle.Utilities)
dtle.Utilities = le.fit_transform(dtle.Utilities)
dfle.LotConfig = le.fit_transform(dfle.LotConfig)
dtle.LotConfig = le.fit_transform(dtle.LotConfig)
dfle.LandSlope = le.fit_transform(dfle.LandSlope)
dtle.LandSlope = le.fit_transform(dtle.LandSlope)
dfle.Neighborhood = le.fit_transform(dfle.Neighborhood)
dtle.Neighborhood = le.fit_transform(dtle.Neighborhood)
dfle.Condition1 = le.fit_transform(dfle.Condition1)
dtle.Condition1 = le.fit_transform(dtle.Condition1)
dfle.Condition2 = le.fit_transform(dfle.Condition2)
dtle.Condition2 = le.fit_transform(dtle.Condition2)
dfle.BldgType = le.fit_transform(dfle.BldgType)
dtle.BldgType = le.fit_transform(dtle.BldgType)
dfle.HouseStyle = le.fit_transform(dfle.HouseStyle)
dtle.HouseStyle = le.fit_transform(dtle.HouseStyle)
dfle.RoofStyle = le.fit_transform(dfle.RoofStyle)
dtle.RoofStyle = le.fit_transform(dtle.RoofStyle)
dfle.RoofMatl = le.fit_transform(dfle.RoofMatl)
dtle.RoofMatl = le.fit_transform(dtle.RoofMatl)
dfle.Exterior1st = le.fit_transform(dfle.Exterior1st)
dtle.Exterior1st = le.fit_transform(dtle.Exterior1st)
dfle.Exterior2nd = le.fit_transform(dfle.Exterior2nd)
dtle.Exterior2nd = le.fit_transform(dtle.Exterior2nd)
dfle.MasVnrType = le.fit_transform(dfle.MasVnrType)
dtle.MasVnrType = le.fit_transform(dtle.MasVnrType)
dfle.ExterQual = le.fit_transform(dfle.ExterQual)
dtle.ExterQual = le.fit_transform(dtle.ExterQual)
dfle.ExterCond = le.fit_transform(dfle.ExterCond)
dtle.ExterCond = le.fit_transform(dtle.ExterCond)
dfle.Foundation = le.fit_transform(dfle.Foundation)
dtle.Foundation = le.fit_transform(dtle.Foundation)
dfle.BsmtQual = le.fit_transform(dfle.BsmtQual)
dtle.BsmtQual = le.fit_transform(dtle.BsmtQual)
dfle.BsmtCond = le.fit_transform(dfle.BsmtCond)
dtle.BsmtCond = le.fit_transform(dtle.BsmtCond)
dfle.BsmtExposure = le.fit_transform(dfle.BsmtExposure)
dtle.BsmtExposure = le.fit_transform(dtle.BsmtExposure)
dfle.BsmtFinType1 = le.fit_transform(dfle.BsmtFinType1)
dtle.BsmtFinType1 = le.fit_transform(dtle.BsmtFinType1)
dfle.BsmtFinType2 = le.fit_transform(dfle.BsmtFinType2)
dtle.BsmtFinType2 = le.fit_transform(dtle.BsmtFinType2)
dfle.Heating = le.fit_transform(dfle.Heating)
dtle.Heating = le.fit_transform(dtle.Heating)
dfle.HeatingQC = le.fit_transform(dfle.HeatingQC)
dtle.HeatingQC = le.fit_transform(dtle.HeatingQC)
dfle.CentralAir = le.fit_transform(dfle.CentralAir)
dtle.CentralAir = le.fit_transform(dtle.CentralAir)
dfle.Electrical = le.fit_transform(dfle.Electrical)
dtle.Electrical = le.fit_transform(dtle.Electrical)
dfle.KitchenQual = le.fit_transform(dfle.KitchenQual)
dtle.KitchenQual = le.fit_transform(dtle.KitchenQual)
dfle.Functional = le.fit_transform(dfle.Functional)
dtle.Functional = le.fit_transform(dtle.Functional)
dfle.FireplaceQu = le.fit_transform(dfle.FireplaceQu)
dtle.FireplaceQu = le.fit_transform(dtle.FireplaceQu)
dfle.GarageType = le.fit_transform(dfle.GarageType)
dtle.GarageType = le.fit_transform(dtle.GarageType)
dfle.GarageFinish = le.fit_transform(dfle.GarageFinish)
dtle.GarageFinish = le.fit_transform(dtle.GarageFinish)
dfle.GarageQual = le.fit_transform(dfle.GarageQual)
dtle.GarageQual = le.fit_transform(dtle.GarageQual)
dfle.GarageCond = le.fit_transform(dfle.GarageCond)
dtle.GarageCond = le.fit_transform(dtle.GarageCond)
dfle.PavedDrive = le.fit_transform(dfle.PavedDrive)
dtle.PavedDrive = le.fit_transform(dtle.PavedDrive)
dfle.PoolQC = le.fit_transform(dfle.PoolQC)
dtle.PoolQC = le.fit_transform(dtle.PoolQC)
dfle.Fence = le.fit_transform(dfle.Fence)
dtle.Fence = le.fit_transform(dtle.Fence)
dfle.MiscFeature = le.fit_transform(dfle.MiscFeature)
dtle.MiscFeature = le.fit_transform(dtle.MiscFeature)
dfle.SaleType = le.fit_transform(dfle.SaleType)
dtle.SaleType = le.fit_transform(dtle.SaleType)
dfle.SaleCondition = le.fit_transform(dfle.SaleCondition)
dtle.SaleCondition = le.fit_transform(dtle.SaleCondition)
'''

#dfle.info()
'''
#histogram to display SalePrice and model probability distribution
df['SalePrice'].describe().transpose()
sns.displot(df['SalePrice'], kde=True, bins=50)
plt.show()
'''

print('Sale Price skewness:',df['SalePrice'].skew())

X_train = dfle.drop(['SalePrice'],axis=1)
X_test = dtle
Y_train = dfle.SalePrice
ohe = OneHotEncoder(drop='first', sparse=False)
dfle[['SaleType','SaleCondition']] = ohe.fit(dfle[['SaleType','SaleCondition']])
dtle[['SaleType','SaleCondition']] = ohe.fit(dtle[['SaleType','SaleCondition']])
dfle.info()
X_train = ohe.fit(X_train[['SaleType','SaleCondition']])
X_test = ohe.fit(X_test[['SaleType','SaleCondition']])
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.3, random_state=101)
lr = LinearRegression()
lr.fit(X_train,Y_train)

Y_predict = lr.predict(X_val)
np.sqrt(mean_squared_error(Y_val, Y_predict))

col_names.remove('SalePrice')
X_test = dtle[col_names]
Y_test = lr.predict(X_test)

output = pd.DataFrame({'Id':dtle['Id'], 'SalePrice':Y_test})
output.to_csv('linreg_discrete.csv', index=False)
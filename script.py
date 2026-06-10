import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso
from sklearn.metrics import root_mean_squared_error
from sklearn.svm import LinearSVR,SVR


train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

#EDA
'''
print(train.info()) # 38 number cols and 43 object cols
print(train.head())

missing=train.isnull().sum()
missing[missing>0].sort_values(ascending=False)
'''

#fill values
#fill object columns with NaN values
obj_cols=train.select_dtypes(include=['object'])
obj_cols=list(obj_cols.columns[obj_cols.isnull().sum()>0])
for col in obj_cols:
    train[col]=train[col].fillna('None')
    test[col] = test[col].fillna('None')

#fill missing numeric values
train['LotFrontage']=train['LotFrontage'].fillna(train['LotFrontage'].median())
train['Electrical']=train['Electrical'].fillna(train['Electrical'].mode()[0])
train['GarageYrBlt'] = train['GarageYrBlt'].fillna(train['YearBuilt'])
train['MasVnrArea']=train['MasVnrArea'].fillna(0)

test['LotFrontage']=test['LotFrontage'].fillna(test['LotFrontage'].median())
test['Electrical']=test['Electrical'].fillna(test['Electrical'].mode()[0])
test['GarageYrBlt'] = test['GarageYrBlt'].fillna(test['YearBuilt'])
test['MasVnrArea']=test['MasVnrArea'].fillna(0)

#encode cat data
ordinal_quality_cols = ['ExterQual', 'ExterCond','BsmtQual', 'BsmtCond','HeatingQC',
                        'KitchenQual','FireplaceQu', 'GarageQual', 'GarageCond']
ordinal_basement_cols = ['BsmtExposure','BsmtFinType1','BsmtFinType2']

quality_map = {'None': 0,'Po': 1,'Fa': 2,'TA': 3,'Gd': 4,'Ex': 5}
bsmt_exposure_map = {'None': 0,'No': 0,'Mn': 1,'Av': 2,'Gd': 3}
bsmt_fin_map = {'None': 0,'Unf': 0,'LwQ': 1,'Rec': 2,'BLQ': 3,'ALQ': 4,'GLQ': 5}

for col in ordinal_quality_cols:
    train[col] = train[col].map(quality_map)
    test[col] = test[col].map(quality_map)

for col in ordinal_basement_cols:
    if col == 'BsmtExposure':
        train[col] = train[col].map(bsmt_exposure_map)
        test[col] = test[col].map(bsmt_exposure_map)
    else:
        train[col] = train[col].map(bsmt_fin_map)
        test[col] = test[col].map(bsmt_fin_map)



#feature adding to both train and test dataset
train['HouseAge']=train['YrSold']-train['YearBuilt']
test['HouseAge']=test['YrSold']-test['YearBuilt']
train['RemodelAge'] = train['YrSold'] - train['YearRemodAdd']
test['RemodelAge'] = test['YrSold'] - test['YearRemodAdd']
train['Qual_Liv'] = train['OverallQual'] * train['GrLivArea']
test['Qual_Liv'] = test['OverallQual'] * test['GrLivArea']
train['TotalSF'] = train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF']
test['TotalSF'] = test['TotalBsmtSF'] + test['1stFlrSF'] + test['2ndFlrSF']
train['TotalBaths'] = (
    train['FullBath']
    + 0.5 * train['HalfBath']
    + train['BsmtFullBath']
    + 0.5 * train['BsmtHalfBath']
)

test['TotalBaths'] = (
    test['FullBath']
    + 0.5 * test['HalfBath']
    + test['BsmtFullBath']
    + 0.5 * test['BsmtHalfBath']
)

#
X=train.drop(columns=['Id','SalePrice','GrLivArea'])
y=train['SalePrice']
y=np.log1p(train['SalePrice'])    # since y is right skewed use log1p



X_train,X_valid,y_train,y_valid=train_test_split(X,y,test_size=0.2,random_state=42)
ordinal_cols=ordinal_quality_cols+ordinal_basement_cols
nominal_cols=[col for col in X_train.select_dtypes(include=['object'])  if col not in ordinal_cols]
num_cols=[col for col in X_train.select_dtypes(include=['number']).columns if col not in ordinal_cols]


preprocessor=ColumnTransformer(
    transformers=[('cat',OneHotEncoder(handle_unknown='ignore'),nominal_cols),
                  ('ord','passthrough',ordinal_cols),
    ('num',StandardScaler(),num_cols)]
)
preprocessor.fit(X_train)
X_train_e=preprocessor.transform(X_train)
X_valid_e=preprocessor.transform(X_valid)




#model=Lasso(alpha=0.0001323)
model=SVR(kernel="poly", degree=2, C=25, epsilon=0)
model.fit(X_train_e,y_train)
preds=model.predict(X_valid_e)
rmse=root_mean_squared_error(y_valid,preds)
print(rmse)










#Benchmarks:
#Simple onehotencoding+linearregression rmse:0.1385
#onehotencoding+ridge(1.0) rmse:0.14
#ordinal+onehot+ridge(1.0) rmse:0.1286
#above with ridge(0.05) and additional features rmse:0.1278
#same but with lasso(0.0001) rmse:0.1210
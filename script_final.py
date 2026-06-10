import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# fill object columns with NaN values
obj_cols = train.select_dtypes(include=['object'])
obj_cols = list(obj_cols.columns[obj_cols.isnull().sum() > 0])

for col in obj_cols:
    train[col] = train[col].fillna('None')
    test[col] = test[col].fillna('None')

# fill missing numeric values
train['LotFrontage'] = train['LotFrontage'].fillna(train['LotFrontage'].median())
train['Electrical'] = train['Electrical'].fillna(train['Electrical'].mode()[0])
train['GarageYrBlt'] = train['GarageYrBlt'].fillna(train['YearBuilt'])
train['MasVnrArea'] = train['MasVnrArea'].fillna(0)

test['LotFrontage'] = test['LotFrontage'].fillna(test['LotFrontage'].median())
test['Electrical'] = test['Electrical'].fillna(test['Electrical'].mode()[0])
test['GarageYrBlt'] = test['GarageYrBlt'].fillna(test['YearBuilt'])
test['MasVnrArea'] = test['MasVnrArea'].fillna(0)
# Missing categorical values
test['MSZoning'] = test['MSZoning'].fillna(test['MSZoning'].mode()[0])
test['Utilities'] = test['Utilities'].fillna(test['Utilities'].mode()[0])
test['Exterior1st'] = test['Exterior1st'].fillna(test['Exterior1st'].mode()[0])
test['Exterior2nd'] = test['Exterior2nd'].fillna(test['Exterior2nd'].mode()[0])
test['KitchenQual'] = test['KitchenQual'].fillna(test['KitchenQual'].mode()[0])
test['Functional'] = test['Functional'].fillna(test['Functional'].mode()[0])
test['SaleType'] = test['SaleType'].fillna(test['SaleType'].mode()[0])

# Missing numeric values
test['BsmtFinSF1'] = test['BsmtFinSF1'].fillna(0)
test['BsmtFinSF2'] = test['BsmtFinSF2'].fillna(0)
test['BsmtUnfSF'] = test['BsmtUnfSF'].fillna(0)
test['TotalBsmtSF'] = test['TotalBsmtSF'].fillna(0)

test['BsmtFullBath'] = test['BsmtFullBath'].fillna(0)
test['BsmtHalfBath'] = test['BsmtHalfBath'].fillna(0)

test['GarageCars'] = test['GarageCars'].fillna(test['GarageCars'].median())
test['GarageArea'] = test['GarageArea'].fillna(test['GarageArea'].median())

# ordinal encoding
ordinal_quality_cols = [
    'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
    'HeatingQC', 'KitchenQual', 'FireplaceQu',
    'GarageQual', 'GarageCond'
]

ordinal_basement_cols = [
    'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'
]

quality_map = {
    'None': 0,
    'Po': 1,
    'Fa': 2,
    'TA': 3,
    'Gd': 4,
    'Ex': 5
}

bsmt_exposure_map = {
    'None': 0,
    'No': 0,
    'Mn': 1,
    'Av': 2,
    'Gd': 3
}

bsmt_fin_map = {
    'None': 0,
    'Unf': 0,
    'LwQ': 1,
    'Rec': 2,
    'BLQ': 3,
    'ALQ': 4,
    'GLQ': 5
}

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

# feature engineering
train['HouseAge'] = train['YrSold'] - train['YearBuilt']
test['HouseAge'] = test['YrSold'] - test['YearBuilt']

train['RemodelAge'] = train['YrSold'] - train['YearRemodAdd']
test['RemodelAge'] = test['YrSold'] - test['YearRemodAdd']

train['Qual_Liv'] = train['OverallQual'] * train['GrLivArea']
test['Qual_Liv'] = test['OverallQual'] * test['GrLivArea']

train['TotalSF'] = (
    train['TotalBsmtSF']
    + train['1stFlrSF']
    + train['2ndFlrSF']
)

test['TotalSF'] = (
    test['TotalBsmtSF']
    + test['1stFlrSF']
    + test['2ndFlrSF']
)

# remove TotalBaths if your experiments showed it hurt performance

# training data
X = train.drop(columns=['Id', 'SalePrice', 'GrLivArea'])
y = np.log1p(train['SalePrice'])

# test data
X_test = test.drop(columns=['Id', 'GrLivArea'])

ordinal_cols = ordinal_quality_cols + ordinal_basement_cols

nominal_cols = [
    col for col in X.select_dtypes(include=['object'])
    if col not in ordinal_cols
]

num_cols = [
    col for col in X.select_dtypes(include=['number']).columns
    if col not in ordinal_cols
]

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), nominal_cols),
        ('ord', 'passthrough', ordinal_cols),
        ('num', StandardScaler(), num_cols)
    ]
)

# encode
X_encoded = preprocessor.fit_transform(X)
X_test_encoded = preprocessor.transform(X_test)

# model
model = Lasso(alpha=0.0001323)
model.fit(X_encoded, y)

# predictions
preds_log = model.predict(X_test_encoded)
preds = np.expm1(preds_log)

# submission
submission = pd.DataFrame({
    'Id': test['Id'],
    'SalePrice': preds
})

submission.to_csv('submission.csv', index=False)

print("submission.csv created")
print(submission.head())
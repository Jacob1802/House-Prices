"""SalePrice - the property's sale price in dollars. This is the target variable that you're trying to predict.

MSSubClass: The building class
MSZoning: The general zoning classification
LotFrontage: Linear feet of street connected to property
LotArea: Lot size in square feet
Street: Type of road access
Alley: Type of alley access
LotShape: General shape of property
LandContour: Flatness of the property
Utilities: Type of utilities available
LotConfig: Lot configuration
LandSlope: Slope of property
Neighborhood: Physical locations within Ames city limits
Condition1: Proximity to main road or railroad
Condition2: Proximity to main road or railroad (if a second is present)
BldgType: Type of dwelling
HouseStyle: Style of dwelling
OverallQual: Overall material and finish quality
OverallCond: Overall condition rating

YearBuilt: Original construction date
YearRemodAdd: Remodel date

RoofStyle: Type of roof
RoofMatl: Roof material
Exterior1st: Exterior covering on house
Exterior2nd: Exterior covering on house (if more than one material)
MasVnrType: Masonry veneer type
MasVnrArea: Masonry veneer area in square feet
ExterQual: Exterior material quality
ExterCond: Present condition of the material on the exterior
Foundation: Type of foundation


BsmtQual: Height of the basement
BsmtCond: General condition of the basement
BsmtExposure: Walkout or garden level basement walls
BsmtFinType1: Quality of basement finished area
BsmtFinSF1: Type 1 finished square feet
BsmtFinType2: Quality of second finished area (if present)
BsmtFinSF2: Type 2 finished square feet
BsmtUnfSF: Unfinished square feet of basement area
TotalBsmtSF: Total square feet of basement area


Heating: Type of heating
HeatingQC: Heating quality and condition
CentralAir: Central air conditioning
Electrical: Electrical system                                                         Electrical  HeatingQC   BsmtQual  HeatingQC KitchenQual LowQualFinSF FireplaceQu GarageQual ExterQual OverallQual
1stFlrSF: First Floor square feet
2ndFlrSF: Second floor square feet
LowQualFinSF: Low quality finished square feet (all floors)
GrLivArea: Above grade (ground) living area square feet
BsmtFullBath: Basement full bathrooms
BsmtHalfBath: Basement half bathrooms
FullBath: Full bathrooms above grade
HalfBath: Half baths above grade
Bedroom: Number of bedrooms above basement level
Kitchen: Number of kitchens
KitchenQual: Kitchen quality
TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
Functional: Home functionality rating
Fireplaces: Number of fireplaces
FireplaceQu: Fireplace quality

GarageType: Garage location
GarageYrBlt: Year garage was built
GarageFinish: Interior finish of the garage
GarageCars: Size of garage in car capacity
GarageArea: Size of garage in square feet
GarageQual: Garage quality
GarageCond: Garage condition
PavedDrive: Paved driveway

WoodDeckSF: Wood deck area in square feet
OpenPorchSF: Open porch area in square feet
EnclosedPorch: Enclosed porch area in square feet
3SsnPorch: Three season porch area in square feet
ScreenPorch: Screen porch area in square feet
PoolArea: Pool area in square feet
PoolQC: Pool quality
Fence: Fence quality
MiscFeature: Miscellaneous feature not covered in other categories
MiscVal: $Value of miscellaneous feature
MoSold: Month Sold
YrSold: Year Sold
SaleType: Type of sale
SaleCondition: Condition of sale"""




# bathroosm per sqmtr
# kitchens per sqmtr
# bedrooms per sqmtr

# combine sqft of all levels - useable sqft

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import re
import sys

# Map for features in order of rank
map_ = {
"Utilities" : {"AllPub": 4, "NoSewr": 3, "NoSeWa": 2, "ELO": 1},
"Electrical" : {"SBrkr": 5, "FuseA": 4, "FuseF": 3, "FuseP": 2, "Mix": 1},
"KitchenQual" : {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1},
"ExterQual" : {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1},
"PoolQC" : {"Ex": 4, "Gd": 3, "TA": 2, "Fa": 1, "Na": 0},
"BsmtQual" : {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "Na": 0},
"FireplaceQu" : {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "Na": 0},
"GarageQual" : {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "Na": 0},
"BsmtFinType1" : {"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1, "Na": 0},
"BsmtFinType2" : {"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1, "Na": 0},
"Fence" : {"GdPrv": 4, "MnPrv": 3, "GdWo": 2, "MnWw": 1, "Na": 0},
"HeatingQC" : {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1},

"ExterCond" : {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1},
"BsmtCond" : {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "Na": 0},
"GarageCond" : {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "Na": 0},
}

qual_features = ['PoolQC', 'BsmtQual', 'FireplaceQu', 'GarageQual', 'BsmtFinType1', 'BsmtFinType2', 'Fence', 'Utilities', 'Electrical', 'HeatingQC', 'KitchenQual', 'ExterQual']
cond_features = ["BsmtCond", "GarageCond", "HeatingQC", "ExterCond"]

foo = ["TotalBsmtSF", "BsmtUnfSF", "BsmtFinSF1", "BsmtFinSF2", "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "WoodDeckSF", "OpenPorchSF"]

def main():
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")

    train_df['dataset'] = "train"
    test_df['dataset'] = "test"


    joined_df = pd.concat([train_df, test_df])

    clean_data(joined_df)

    qualcond_percent(joined_df, qual_features + qual_features, "Qual")
    qualcond_percent(joined_df, cond_features, "Cond")
    
    print(joined_df[f'QualPercen'])
    print(joined_df[f'CondPercen'])
    # replace na vals in series in prep for bining


def clean_data(df):
    df['Utilities'] = df['Utilities'].fillna(df['Utilities'].mode()[0])
    df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])
    df['KitchenQual'] = df['KitchenQual'].fillna(df['KitchenQual'].mode()[0])
    
    
    df[qual_features] = df[qual_features].fillna('Na')
    df[cond_features] = df[cond_features].fillna('Na')


def qualcond_percent(df, features, type):
    """
    Calc percentage score for Quality and Condition
    """
    df[f'{type}Max'] = 0
    df[f'{type}Total'] = 0
    # sum up feature ratings to find the total score of said house and max score
    for feature in features:
        df[f'{feature}_rating'] = df[feature].apply(lambda x: map_[feature][x])
        df[f'{type}Max'] += df[f'{feature}_rating'].apply(lambda x: x if x == 0 else df[f'{feature}_rating'].unique().max())
        df[f'{type}Total'] += df[f'{feature}_rating']


    df[f'{type}Percen'] = round(df[f'{type}Total'] / df[f'{type}Max'], 2)


def root_mean_squared_logarithmic_error(y_true, y_pred):
    """
    Compute the Root Mean Squared Logarithmic Error between two arrays.
    """
    rmsle = mean_squared_error(y_true, y_pred, squared=False)
    return rmsle


if __name__ == "__main__":
    main()
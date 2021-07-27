import pandas as pd
import numpy as np

data=pd.read_csv('data.csv')

print(data.isnull().sum())
data = data.dropna(how='any',axis=0)
print(data.isnull().sum())

columnsX=data.columns[2:-1]
columnsY=data.columns[-1]

from sklearn.model_selection import KFold

catCol=['Gender','Married','Self_Employed','Region','Dependents']
numWithScalingCol=['Salary','Loan_Amount','Loan_Amount_Term']
numerCol=['Credit_History']

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn import ensemble
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
import joblib

numeric_transformer_scaling = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value=99))])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer_scaling, numWithScalingCol),
        ('num2', numeric_transformer, numerCol),
        ('cat', categorical_transformer, catCol)])


modelRandomForest = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', ensemble.RandomForestClassifier())])

modelRandomForest.fit(data[columnsX],data[columnsY]);

joblib.dump(modelRandomForest,'modelRandomForest.pkl')

modelDecisionTree = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', tree.DecisionTreeClassifier())])

modelDecisionTree.fit(data[columnsX],data[columnsY]);

joblib.dump(modelDecisionTree,'modelDecisionTree.pkl')

modelGaussianNB = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', GaussianNB())])

modelGaussianNB.fit(data[columnsX],data[columnsY]);

predicted=modelGaussianNB.predict(data[columnsX])

joblib.dump(modelGaussianNB,'modelGaussianNB.pkl')
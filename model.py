import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor

df = pd.read_csv('df_to_deploy.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)

X = df.drop('Price', axis=1)
print(X.head())

y = df['Price']

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=50)

cb = CatBoostRegressor()
cb.fit(xtrain, ytrain)

ypred = cb.predict(xtest)

# USE PICKLE TO SAVE MODEL TO DISK

pickle.dump(cb, open('FlightPricePredictmodel.pkl', 'wb'))

print(ypred)

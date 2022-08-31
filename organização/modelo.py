# imports necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit,cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures,StandardScaler,MinMaxScaler
from sklearn.metrics import mean_squared_error,make_scorer


# loads dataset
df = pd.read_csv("data/train.csv")
df2 = pd.read_csv("data/additive.csv")


data_train = df[["DT_COMPTC"]][:3000]

#dados de treino
X_train = df.drop(["Fluxo","DT_COMPTC"],axis=1)[:3000]
y_train = df[["Fluxo"]][:3000]

#não sei bem pra que isso mas ajuda
data_test = df[["DT_COMPTC"]][3000:]

#dados de teste
X_test = df.drop(["Fluxo","DT_COMPTC"],axis=1)[3000:]
y_test = df[["Fluxo"]][3000:]

#preservamos os dados
df['RESG_DIA'] = df2['RESG_DIA'].copy()
df["CAPTC_DIA"] = df2["CAPTC_DIA"].copy()

#vemos correlação de colunas
# df.corr()[['RESG_DIA', "CAPTC_DIA"]]

#padroniza os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
y_train = scaler.fit_transform(y_train)
X_test = scaler.fit_transform(X_test)
y_test = scaler.fit_transform(y_test)

tscv = TimeSeriesSplit()
# cv_results = cross_val_score(RandomForestRegressor(500),X_train,y_train,cv=tscv,scoring=make_scorer(mean_squared_error))

# print(cv_results)

model = RandomForestRegressor(1000)

model.fit(X_train,y_train)

resposta = model.predict(X_test)

plt.plot(data_test,y_test)
plt.plot(data_test,resposta)
plt.show()


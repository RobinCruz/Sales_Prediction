import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as lr
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.metrics import r2_score as r2
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split
import seaborn as sns
import pickle

df = pd.read_csv("train.csv")
tdf = pd.read_csv("test.csv")

year = list()
month = list()
date = list()

tyear = list()
tmonth = list()
tdate = list()

for i in df["date"]:
	Y,M,D = tuple(map(int,i.split("-")))
	year.append(Y)
	month.append(M)
	date.append(D)

for i in tdf["date"]:
	Y,M,D = tuple(map(int,i.split("-")))
	tyear.append(Y)
	tmonth.append(M)
	tdate.append(D)

df = df.drop("date",axis=1)
tdf = tdf.drop("date",axis=1)

df.insert(0,"year",year,True)
df.insert(1,"month",month,True)
df.insert(2,"date",date,True)

tdf.insert(0,"year",tyear,True)
tdf.insert(1,"month",tmonth,True)
tdf.insert(2,"date",tdate,True)

X_train = df[["store","month","day","item"]]
Y_train = df["sales"]

Mean = X_train.mean()
MeanY = Y_train.mean()

X_train["store"] = (X_train["store"]-Mean[0])/Mean[0]
X_train["month"] = (X_train["month"]-Mean[1])/Mean[1]
X_train["day"] = (X_train["day"]-Mean[2])/Mean[2]
X_train["item"] = (X_train["item"]-Mean[3])/Mean[3]
Y_train = (Y_train-MeanY)/MeanY

X = np.asanyarray(X_train)
Y = np.asanyarray(Y_train)

#X_test = df[["store","month","day","item"]]
#Y_test = tdf["sales"]

#X_test = np.asanyarray(X_test)
#Y_test = np.asanyarray(Y)
#print(X)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=20)

#model = lr()
print(Y_train)
print("model starts")
model = rfr(max_features = "log2",n_estimators=6)
model.fit(X_train,Y_train)
print("model ends")
Pred = model.predict(X_test)
pickle.dump(model, open("RandomForestRegressor.sav","wb"))

model = pickle.load(open("RandomForestRegressor.sav","rb"))
'''
Pred1 = pd.DataFrame()
Pred1.insert(0,"sales",Pred,True)
Pred1.insert(0,"id",tdf["id"],True)
Pred1.to_csv("pred.csv", index=False)
'''

print("r2_score : %2f"%r2(Y_test,Pred))
print("MAE		: %2f"%mae(Y_test,Pred))
print("MSE 		: %2f"%mse(Y_test,Pred))

'''
plt.scatter([i for i in range(len(Pred))],Pred,color="red")
plt.show()
plt.scatter([i for i in range(len(Y_test))],Y_test,color="green")
plt.show()

plt.scatter(df["day"],df["sales"])
plt.show()
'''
Sdf = pd.DataFrame()
Sdf.insert(0,"Actual",Y_test,True)
Sdf.insert(1,"Pred",Pred,True)

sns.kdeplot(data=Sdf,x="Actual")

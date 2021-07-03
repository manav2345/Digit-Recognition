#%%
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score
from sklearn.ensemble import  RandomForestClassifier
Train=pd.read_csv("train.csv")
Train.head()
#%%
import seaborn as sns
sns.countplot(Train["label"])
#%%
x=Train.drop("label",axis=1)
y=Train["label"]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=1)
random_forest = RandomForestClassifier(n_estimators=100,random_state=15)
random_forest.fit(x_train, y_train)
pri = random_forest.predict(x_test)
print("Random forest\n",r2_score(y_test,pri)*100)
#%%
Test=pd.read_csv("test.csv")
sample_submission=pd.read_csv("sample_submission .csv")
x=Train.drop("label",axis=1)
y=Train["label"]
random_forest = RandomForestClassifier(n_estimators=100,random_state=15)
random_forest.fit(x,y)
pri = random_forest.predict(Test)
print("Random forest\n",r2_score(sample_submission["Label"],pri)*100)

#%%

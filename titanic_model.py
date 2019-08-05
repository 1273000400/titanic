#-*-coding:utf-8-*-
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier,AdaBoostRegressor
from sklearn.metrics import mean_squared_error, explained_variance_score
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
train = pd.read_csv('template/train.csv')
test = pd.read_csv('template/test.csv')


# print(train.info())

ava_list = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']

x_train = train[ava_list]
x_test = test[ava_list]


y_train=train[['Survived']]
# print(x_train.info())
# print()


x_train['Age'].fillna(x_train['Age'].mean(),inplace=True)
x_train['Embarked'].fillna('S',inplace=True)
x_train['Fare'].fillna(x_train['Fare'].mean(),inplace=True)

x_test['Age'].fillna(x_test['Age'].mean(),inplace=True)
# x_test['Embarked'].fillna('S',inplace=True)
x_test['Fare'].fillna(x_test['Fare'].mean(),inplace=True)


#特征向量化
dict_vec = DictVectorizer(sparse=False)
x_train = dict_vec.fit_transform(x_train.to_dict(orient='record'))
x_test = dict_vec.fit_transform(x_test.to_dict(orient='record'))
print(dict_vec.feature_names_)

#--------------------------------------------------------
# num_training = int(0.9 * len(x_train))
# x_train1, y_train1 = x_train[:num_training], y_train[:num_training]
# a_test, b_test = x_train[num_training:], y_train[num_training:]
#---------------------------------------------------------
#分类预测
# rfc = RandomForestClassifier()
# rfc.fit(x_train1,y_train1)
# c_test = rfc.predict(a_test)
# print(cross_val_score(rfc,x_train1,y_train1,cv=5).mean())

# xgbc = XGBClassifier()
# xgbc.fit(x_train1,y_train1)
# c_test = xgbc.predict(a_test)
# print(cross_val_score(xgbc,x_train1,y_train1,cv=5).mean())

# dt_regressor = DecisionTreeRegressor(max_depth=4)
# dt_regressor.fit(x_train1,y_train1)
# c_test = dt_regressor.predict(a_test)
# print(cross_val_score(dt_regressor,x_train1,y_train1,cv=5).mean())

# dt_regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=400, random_state=7)
# dt_regressor.fit(x_train1,y_train1)
# c_test = dt_regressor.predict(a_test)
# print(cross_val_score(dt_regressor,x_train1,y_train1,cv=5).mean())

# mse = mean_squared_error(b_test, c_test)
# evs = explained_variance_score(b_test, c_test)
#
# print ("Mean squared error =", round(mse, 2))
# print ("Explained variance score =", round(evs, 2))

#-------------------------------------------------------------
# rfc = RandomForestClassifier()
# rfc.fit(x_train,y_train)
# c_test = rfc.predict(x_test)
# df = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':c_test})
# df.to_csv('rfc_submission.csv',index=False)

# xgbc = XGBClassifier()
# xgbc.fit(x_train,y_train)
# c_test = xgbc.predict(x_test)
# df = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':c_test})
# df.to_csv('xgbc_submission.csv',index=False)

# dtr = DecisionTreeRegressor(max_depth=4)
# dtr.fit(x_train,y_train)
# c_test = dtr.predict(x_test)
# df = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':[1 if x>=0.5 else 0 for x in c_test]})
# df.to_csv('dtr_submission.csv',index=False)

ads_dtr = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=400, random_state=7)
ads_dtr.fit(x_train,y_train)
c_test = ads_dtr.predict(x_test)
df = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':[1 if x>=0.5 else 0 for x in c_test]})
df.to_csv('ads_dtr_submission.csv',index=False)
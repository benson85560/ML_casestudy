#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib import style
from glob import glob
from sklearn import preprocessing


# In[2]:


# total runtime : 2:42

df1 = pd.read_excel('00_業務員資料.xlsx', sheet_name = 1) # No yyyymm
df2 = pd.read_excel('01_FLI_業務員登入次數.xlsx', sheet_name = 1)
df3 = pd.read_excel('02_FLI_業務員定聯紀錄.xlsx', sheet_name = 1)
df4 = pd.read_excel('03_FLI_業務員約訪次數.xlsx', sheet_name = 1)
df5 = pd.read_excel('04_FLI_業務員擁有客戶數_memo字數.xlsx', sheet_name = 1) # No yyyymm
df6 = pd.read_excel('05_問卷資料.xlsx', sheet_name = 1)
df7 = pd.read_excel('06_保單健檢.xlsx', sheet_name = 1)
df8 = pd.read_excel('07_壽險FYC.xlsx', sheet_name = 1)
df9 = pd.read_excel('08_產險FYC.xlsx', sheet_name = 1)


# In[3]:


df_list = [df2, df3, df4, df6, df7, df8, df9] 
df = reduce(lambda  left,right: pd.merge(left,right,on=['AGENT_ID', 'YYYYMM'], how='inner'), df_list)

df = df.groupby(['AGENT_ID']).sum().reset_index()
df.drop(['YYYYMM'], axis = 1, inplace = True)

df_list = [df1, df5, df]
df = reduce(lambda  left,right: pd.merge(left,right,on=['AGENT_ID'], how='inner'), df_list)


# In[4]:


# Drop na row
df.dropna(subset=["LOCATION_ID"], inplace=True)
df.reset_index(inplace=True)


# In[5]:


# 把 START_DATE, END_DATE, BIR_DATE 刪掉
df = df.drop(['START_DATE', 'END_DATE', 'BIR_DATE'], axis = 1)
# drop agent_id
df = df.drop(['AGENT_ID'], axis = 1)


# In[6]:


# 'UNIT_POST_CODE', 'CITY', 'DISTRICT', 'LOCATION_ID' : 都做 frequency encoding，這裡就沒特別看數目有無重複，有的話應該是郵遞區號，目前先忽略。
unit_post_code_encode = df.groupby('UNIT_POST_CODE').size() / len(df)
city_encode = df.groupby('CITY').size() / len(df)
district_encode = df.groupby('DISTRICT').size() / len(df)
location_id_encode = df.groupby('LOCATION_ID').size() / len(df)

df['UNIT_POST_CODE_encode'] = df['UNIT_POST_CODE'].apply(lambda x : unit_post_code_encode[x])
df['CITY_encode'] = df['CITY'].apply(lambda x : city_encode[x])
df['DISTRICT_encode'] = df['DISTRICT'].apply(lambda x : district_encode[x])
df['LOCATION_ID_encode'] = df['LOCATION_ID'].apply(lambda x : location_id_encode[x])


# In[7]:


# AGENT_TITLE : frequency encoding，有發現 frequency 重複的問題，這裡先忽略，原因是先假設差異有，但是不大。
agent_title_encode = df.groupby('AGENT_TITLE').size() / len(df)
df['AGENT_TITLE_encode'] = df['AGENT_TITLE'].apply(lambda x : agent_title_encode[x])


# In[8]:


df


# In[9]:


# 標準化數值變數
num_vars = ['LOGIN_CNT', 'CONTACT_CNT', 'CONTACT_CUST_CNT', 'VISIT_CNT', 'CUSTOMER_VISIT_CNT', 'CUSTOMER_ID_CNT', 'MEMO_LENGTH', 'QNR_CNT', 'CUSTOMER_CNT', 'CONSULT_CNT', 'CONSULT_CUST_CNT']
for var in num_vars :
  temp = pd.Series(preprocessing.scale(df[var]))
  temp.name = str(var) + '_normal'
  temp.to_frame()
  df = pd.concat([df, temp], axis = 1)


# In[10]:


# 處理 Y 變數 : Premium Sum
df['premium_sum'] = df['PREMIUM'] + df['PROPERTY_INSURANCE_PERMIUM']
df['premium_sum_log'] = np.log(df['PREMIUM'] + df['PROPERTY_INSURANCE_PERMIUM'])


# In[11]:


df.columns


# In[12]:


len(df.columns)


# In[13]:


# 抓出訓練用 DF
# 標準化
df_ML = df[['ISABLE', 'AGENT_AGE', 'ON_BOARD_AGE', 'SENIORITY', 'AGENT_SEX', 'AGENT_TITLE_encode', 'UNIT_POST_CODE_encode', 'CITY_encode', 'DISTRICT_encode',
             'LOCATION_ID_encode', 'LOGIN_CNT_normal', 'CONTACT_CNT_normal', 'CONTACT_CUST_CNT_normal', 'VISIT_CNT_normal', 'CUSTOMER_VISIT_CNT_normal', 'CUSTOMER_ID_CNT_normal',
             'MEMO_LENGTH_normal', 'QNR_CNT_normal', 'CUSTOMER_CNT_normal', 'CONSULT_CNT_normal', 'CONSULT_CUST_CNT_normal', 'premium_sum', 'premium_sum_log']]
# 正規化
# fubon_df_ML = fubon_df[['ISABLE', 'AGENT_AGE', 'ON_BOARD_AGE', 'SENIORITY', 'AGENT_SEX', 'AGENT_TITLE_encode', 'UNIT_POST_CODE_encode', 'CITY_encode', 'DISTRICT_encode',
#                         'LOGIN_CNT', 'CONTACT_CNT', 'CONTACT_CUST_CNT', 'VISIT_CNT', 'CUSTOMER_VISIT_CNT', 'CUSTOMER_ID_CNT', 'MEMO_LENGTH', 'QNR_CNT', 'CUSTOMER_CNT', 'CONSULT_CNT', 'CONSULT_CUST_CNT',
#              'LOCATION_ID_encode', 'premium_sum', 'premium_sum_log']]

df_ML


# In[14]:


df_ML.info()


# # Corr

# In[15]:


# find corr

corr_matrix = round(df_ML.corr(), 3)
corr_matrix['premium_sum_log'].sort_values(ascending = False).head(20)


# In[16]:


from pandas.plotting import scatter_matrix

attributes = ["premium_sum_log", "CUSTOMER_ID_CNT_normal", "SENIORITY", 'CONSULT_CUST_CNT_normal', 'CUSTOMER_VISIT_CNT_normal']
scatter_matrix(df_ML[attributes], figsize=(18, 14))
plt.show()


# In[17]:


# Further Insights
df_ML.plot(kind = 'scatter', x = 'SENIORITY', y = 'CUSTOMER_ID_CNT_normal', figsize = (5, 3))


# # SVR

# In[18]:


from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression


# In[19]:


y = df_ML['premium_sum_log']
X = df_ML[['ISABLE', 'AGENT_AGE', 'ON_BOARD_AGE', 'SENIORITY', 'AGENT_SEX', 'AGENT_TITLE_encode', 'UNIT_POST_CODE_encode', 'CITY_encode', 'DISTRICT_encode',
             'LOCATION_ID_encode', 'LOGIN_CNT_normal', 'CONTACT_CNT_normal', 'CONTACT_CUST_CNT_normal', 'VISIT_CNT_normal', 'CUSTOMER_VISIT_CNT_normal', 'CUSTOMER_ID_CNT_normal',
             'MEMO_LENGTH_normal', 'QNR_CNT_normal', 'CUSTOMER_CNT_normal', 'CONSULT_CNT_normal', 'CONSULT_CUST_CNT_normal']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train[:3]


# In[20]:


svr_model = SVR(C=3, cache_size=500, epsilon=1, kernel='rbf')

# Train the model 
svr_model.fit(X_train, y_train)


# In[21]:


predicted_values = svr_model.predict(X_test)


# In[22]:


from sklearn.metrics import mean_squared_error, r2_score

print("Mean Squared Error = ", round(mean_squared_error(predicted_values, y_test), 3))
print("Root Mean Squared Error = ", round(np.sqrt(mean_squared_error(predicted_values, y_test)), 3))


# # KNN-reg

# In[23]:


y = df_ML['premium_sum_log']
X = df_ML[['ISABLE', 'AGENT_AGE', 'ON_BOARD_AGE', 'SENIORITY', 'AGENT_SEX', 'AGENT_TITLE_encode', 'UNIT_POST_CODE_encode', 'CITY_encode', 'DISTRICT_encode',
             'LOCATION_ID_encode', 'LOGIN_CNT_normal', 'CONTACT_CNT_normal', 'CONTACT_CUST_CNT_normal', 'VISIT_CNT_normal', 'CUSTOMER_VISIT_CNT_normal', 'CUSTOMER_ID_CNT_normal',
             'MEMO_LENGTH_normal', 'QNR_CNT_normal', 'CUSTOMER_CNT_normal', 'CONSULT_CNT_normal', 'CONSULT_CUST_CNT_normal']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train[:3]


# In[24]:


from sklearn.neighbors import KNeighborsRegressor

knn_model = KNeighborsRegressor(n_neighbors=5, metric='euclidean').fit(X_train, y_train)
predicted_values = knn_model.predict(X_test)


# In[25]:


predict_df_KNN = pd.DataFrame({"Dependent_Test" : y_test, "Dependent_Predicted" : predicted_values})
predict_df_KNN.head()


# In[26]:


from sklearn.metrics import mean_squared_error, r2_score

print("Mean Squared Error = ", mean_squared_error(predict_df_KNN.Dependent_Predicted, predict_df_KNN.Dependent_Test))
print("Root Mean Squared Error = ", np.sqrt(mean_squared_error(predict_df_KNN.Dependent_Predicted, predict_df_KNN.Dependent_Test)))


# In[27]:


r2_score(predict_df_KNN.Dependent_Test, predict_df_KNN.Dependent_Predicted)


# # Random Forest reg

# In[28]:


y = df_ML['premium_sum_log']
X = df_ML[['ISABLE', 'AGENT_AGE', 'ON_BOARD_AGE', 'SENIORITY', 'AGENT_SEX', 'AGENT_TITLE_encode', 'UNIT_POST_CODE_encode', 'CITY_encode', 'DISTRICT_encode',
             'LOCATION_ID_encode', 'LOGIN_CNT_normal', 'CONTACT_CNT_normal', 'CONTACT_CUST_CNT_normal', 'VISIT_CNT_normal', 'CUSTOMER_VISIT_CNT_normal', 'CUSTOMER_ID_CNT_normal',
             'MEMO_LENGTH_normal', 'QNR_CNT_normal', 'CUSTOMER_CNT_normal', 'CONSULT_CNT_normal', 'CONSULT_CUST_CNT_normal']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[29]:


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=50)
rf.fit(X_train, y_train)


# In[30]:


plt.figure(figsize = (15, 15))
plt.barh(X.columns, rf.feature_importances_)
plt.title('random forest result', size = 20)


# In[31]:


predicted_values = rf.predict(X_test)

print("Mean Squared Error = ", round(mean_squared_error(predicted_values, y_test), 3))
print("Root Mean Squared Error = ", round(np.sqrt(mean_squared_error(predicted_values, y_test)), 3))


# # Adaboost

# In[32]:


from sklearn.ensemble import AdaBoostRegressor
from sklearn import metrics


# In[33]:


y = df_ML['premium_sum_log']
X = df_ML[['ISABLE', 'AGENT_AGE', 'ON_BOARD_AGE', 'SENIORITY', 'AGENT_SEX', 'AGENT_TITLE_encode', 'UNIT_POST_CODE_encode', 'CITY_encode', 'DISTRICT_encode',
             'LOCATION_ID_encode', 'LOGIN_CNT_normal', 'CONTACT_CNT_normal', 'CONTACT_CUST_CNT_normal', 'VISIT_CNT_normal', 'CUSTOMER_VISIT_CNT_normal', 'CUSTOMER_ID_CNT_normal',
             'MEMO_LENGTH_normal', 'QNR_CNT_normal', 'CUSTOMER_CNT_normal', 'CONSULT_CNT_normal', 'CONSULT_CUST_CNT_normal']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[34]:


adaboost_regressor = AdaBoostRegressor(n_estimators=1500, learning_rate = 0.001, loss='exponential')
ada_model = adaboost_regressor.fit(X_train, y_train)
predicted_vaAlues = ada_model.predict(X_test)


# In[35]:


print("Mean Squared Error = ", round(mean_squared_error(predicted_values, y_test), 3))
print("Root Mean Squared Error = ", round(np.sqrt(mean_squared_error(predicted_values, y_test)), 3))


# # grouping

# In[36]:


group1 = ['AG', 'CA']
group2 = ['MS', 'MAM', 'MM', 'SP', 'AM', 'UM']
group3 = ['SMM', 'DM', 'VRM', 'ARM', 'SRM', 'AVP', 'SVP']
# sum 2&3

# group1 = ['AG', 'CA']
# group2 = ['MS', 'MAM', 'MM', 'SP', 'AM', 'UM', 'SMM', 'DM', 'VRM', 'ARM', 'SRM', 'AVP', 'SVP']

df_1 = df[df['AGENT_TITLE'].isin(group1)] # AG doesn't exist
df_2 = df[df['AGENT_TITLE'].isin(group2)]
df_3 = df[df['AGENT_TITLE'].isin(group3)]
len(df_1),len(df_2), len(df_3)


# ## group1

# In[37]:


ML_df_1 = df_1[['ISABLE', 'AGENT_AGE', 'ON_BOARD_AGE', 'SENIORITY', 'AGENT_SEX', 'UNIT_POST_CODE_encode', 'CITY_encode', 'DISTRICT_encode',
             'LOCATION_ID_encode', 'LOGIN_CNT_normal', 'CONTACT_CNT_normal', 'CONTACT_CUST_CNT_normal', 'VISIT_CNT_normal', 'CUSTOMER_VISIT_CNT_normal', 'CUSTOMER_ID_CNT_normal',
             'MEMO_LENGTH_normal', 'QNR_CNT_normal', 'CUSTOMER_CNT_normal', 'CONSULT_CNT_normal', 'CONSULT_CUST_CNT_normal', 'premium_sum', 'premium_sum_log']]


# In[38]:


y_g1 = ML_df_1['premium_sum_log']
X_g1 = ML_df_1[['ISABLE', 'AGENT_AGE', 'ON_BOARD_AGE', 'SENIORITY', 'AGENT_SEX', 'UNIT_POST_CODE_encode', 'CITY_encode', 'DISTRICT_encode',
             'LOCATION_ID_encode', 'LOGIN_CNT_normal', 'CONTACT_CNT_normal', 'CONTACT_CUST_CNT_normal', 'VISIT_CNT_normal', 'CUSTOMER_VISIT_CNT_normal', 'CUSTOMER_ID_CNT_normal',
             'MEMO_LENGTH_normal', 'QNR_CNT_normal', 'CUSTOMER_CNT_normal', 'CONSULT_CNT_normal', 'CONSULT_CUST_CNT_normal']]

X_train_g1, X_test_g1, y_train_g1, y_test_g1 = train_test_split(X_g1, y_g1, test_size=0.2, random_state=42)


# In[39]:


rf = RandomForestRegressor(n_estimators=50)
rf.fit(X_train_g1, y_train_g1)


# In[40]:


plt.figure(figsize = (15, 15))
plt.barh(X_g1.columns, rf.feature_importances_)
plt.title('group1 result', size = 20)


# In[41]:


predicted_values = rf.predict(X_test_g1)

print("Mean Squared Error = ", round(mean_squared_error(predicted_values, y_test_g1), 3))
print("Root Mean Squared Error = ", round(np.sqrt(mean_squared_error(predicted_values, y_test_g1)), 3))


# ## group2

# In[42]:


ML_df_2 = df_2[['ISABLE', 'AGENT_AGE', 'ON_BOARD_AGE', 'SENIORITY', 'AGENT_SEX', 'UNIT_POST_CODE_encode', 'CITY_encode', 'DISTRICT_encode',
             'LOCATION_ID_encode', 'LOGIN_CNT_normal', 'CONTACT_CNT_normal', 'CONTACT_CUST_CNT_normal', 'VISIT_CNT_normal', 'CUSTOMER_VISIT_CNT_normal', 'CUSTOMER_ID_CNT_normal',
             'MEMO_LENGTH_normal', 'QNR_CNT_normal', 'CUSTOMER_CNT_normal', 'CONSULT_CNT_normal', 'CONSULT_CUST_CNT_normal', 'premium_sum', 'premium_sum_log']]
ML_df_2


# In[43]:


y_g2 = ML_df_2['premium_sum_log']
X_g2 = ML_df_2[['ISABLE', 'AGENT_AGE', 'ON_BOARD_AGE', 'SENIORITY', 'AGENT_SEX', 'UNIT_POST_CODE_encode', 'CITY_encode', 'DISTRICT_encode',
             'LOCATION_ID_encode', 'LOGIN_CNT_normal', 'CONTACT_CNT_normal', 'CONTACT_CUST_CNT_normal', 'VISIT_CNT_normal', 'CUSTOMER_VISIT_CNT_normal', 'CUSTOMER_ID_CNT_normal',
             'MEMO_LENGTH_normal', 'QNR_CNT_normal', 'CUSTOMER_CNT_normal', 'CONSULT_CNT_normal', 'CONSULT_CUST_CNT_normal']]

X_train_g2, X_test_g2, y_train_g2, y_test_g2 = train_test_split(X_g2, y_g2, test_size=0.2, random_state=42)


# In[44]:


rf = RandomForestRegressor(n_estimators=50)
rf.fit(X_train_g2, y_train_g2)


# In[45]:


plt.figure(figsize = (15, 15))
plt.barh(X_g2.columns, rf.feature_importances_)
plt.title('group2 result', size = 20)


# In[46]:


predicted_values = rf.predict(X_test_g2)

print("Mean Squared Error = ", round(mean_squared_error(predicted_values, y_test_g2), 3))
print("Root Mean Squared Error = ", round(np.sqrt(mean_squared_error(predicted_values, y_test_g2)), 3))


# # group3

# In[47]:


ML_df_3 = df_3[['ISABLE', 'AGENT_AGE', 'ON_BOARD_AGE', 'SENIORITY', 'AGENT_SEX', 'UNIT_POST_CODE_encode', 'CITY_encode', 'DISTRICT_encode',
             'LOCATION_ID_encode', 'LOGIN_CNT_normal', 'CONTACT_CNT_normal', 'CONTACT_CUST_CNT_normal', 'VISIT_CNT_normal', 'CUSTOMER_VISIT_CNT_normal', 'CUSTOMER_ID_CNT_normal',
             'MEMO_LENGTH_normal', 'QNR_CNT_normal', 'CUSTOMER_CNT_normal', 'CONSULT_CNT_normal', 'CONSULT_CUST_CNT_normal', 'premium_sum', 'premium_sum_log']]

y_g3 = ML_df_3['premium_sum_log']
X_g3 = ML_df_3[['ISABLE', 'AGENT_AGE', 'ON_BOARD_AGE', 'SENIORITY', 'AGENT_SEX', 'UNIT_POST_CODE_encode', 'CITY_encode', 'DISTRICT_encode',
             'LOCATION_ID_encode', 'LOGIN_CNT_normal', 'CONTACT_CNT_normal', 'CONTACT_CUST_CNT_normal', 'VISIT_CNT_normal', 'CUSTOMER_VISIT_CNT_normal', 'CUSTOMER_ID_CNT_normal',
             'MEMO_LENGTH_normal', 'QNR_CNT_normal', 'CUSTOMER_CNT_normal', 'CONSULT_CNT_normal', 'CONSULT_CUST_CNT_normal']]

X_train_g3, X_test_g3, y_train_g3, y_test_g3 = train_test_split(X_g3, y_g3, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=50)
rf.fit(X_train_g3, y_train_g3)


# In[48]:


plt.figure(figsize = (15, 15))
plt.barh(X_g3.columns, rf.feature_importances_)
plt.title('group3 result', size = 20)


# In[49]:


predicted_values = rf.predict(X_test_g3)

print("Mean Squared Error = ", round(mean_squared_error(predicted_values, y_test_g3), 3))
print("Root Mean Squared Error = ", round(np.sqrt(mean_squared_error(predicted_values, y_test_g3)), 3))


# # linear reg(看數位活動的正負號)

# ## group 1

# In[70]:


# 將y放入x
X_g1['premium_sum_log'] = y_g1


# In[51]:


# X_g1['premium_sum_log'] = y_g1
formula = '{} ~ {} + 1'.format('premium_sum_log', ' + '.join(list(X_g1.columns)[:-1]))
model = smf.ols(formula, X_g1).fit()
model.summary()


# ## group 2

# In[52]:


X_g2['premium_sum_log'] = y_g2
formula = '{} ~ {} + 1'.format('premium_sum_log', ' + '.join(list(X_g2.columns)[:-1]))
model = smf.ols(formula, X_g2).fit()
model.summary()


# ## group 3

# In[53]:


X_g3['premium_sum_log'] = y_g3
formula = '{} ~ {} + 1'.format('premium_sum_log', ' + '.join(list(X_g3.columns)[:-1]))
model = smf.ols(formula, X_g3).fit()
model.summary()


# # appendix : top 10 reg（只測前十具影響力的變數）

# In[72]:


X_g1.drop(columns = 'premium_sum_log', inplace = True)
X_g2.drop(columns = 'premium_sum_log', inplace = True)
X_g3.drop(columns = 'premium_sum_log', inplace = True)


# In[54]:


# group 1


# In[74]:


rf = RandomForestRegressor(n_estimators=50)
rf.fit(X_train_g1, y_train_g1)


# In[75]:



a = {'col': np.round(rf.feature_importances_, 2)}
kk = pd.DataFrame(data = a, index = X_g1.columns)
X_g1_top10 = kk.sort_values(by = ['col'], ascending = False).head(10)
X_g1_top10 = X_g1[X_g1_top10.index]
X_g1_top10['premium_sum_log'] = y_g1.copy()


# In[76]:


formula = '{} ~ {} + 1'.format('premium_sum_log', ' + '.join(list(X_g1_top10.columns[:-1])))
model = smf.ols(formula, X_g1_top10).fit()
model.summary()


# In[77]:


# group2


# In[78]:


rf = RandomForestRegressor(n_estimators=50)
rf.fit(X_train_g2, y_train_g2)


# In[79]:


a = {'col': np.round(rf.feature_importances_, 2)}
kk = pd.DataFrame(data = a, index = X_g2.columns)
X_g2_top10 = kk.sort_values(by = ['col'], ascending = False).head(10)
X_g2_top10 = X_g2[X_g2_top10.index]
X_g2_top10['premium_sum_log'] = y_g2.copy()


# In[80]:


formula = '{} ~ {} + 1'.format('premium_sum_log', ' + '.join(list(X_g2_top10.columns[:-1])))
model = smf.ols(formula, X_g2_top10).fit()
model.summary()


# In[81]:


# group3


# In[82]:


rf.fit(X_train_g3, y_train_g3)


# In[83]:


a = {'col': np.round(rf.feature_importances_, 2)}
kk = pd.DataFrame(data = a, index = X_g3.columns)
X_g3_top10 = kk.sort_values(by = ['col'], ascending = False).head(10)
X_g3_top10 = X_g3[X_g3_top10.index]
X_g3_top10['premium_sum_log'] = y_g3.copy()


# In[84]:


formula = '{} ~ {} + 1'.format('premium_sum_log', ' + '.join(list(X_g3_top10.columns[:-1])))
model = smf.ols(formula, X_g3_top10).fit()
model.summary()


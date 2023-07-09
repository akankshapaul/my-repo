#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(color_codes=True)

df = pd.read_csv("heart_failure_clinical_records_dataset.csv")
df


# In[2]:


df.head()


# In[3]:


df.columns


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


#to detect outliers in the data showing the trend of people having creatinine with increasing age
plt.xlabel("Age")
plt.ylabel("Creatinine phosphokinase")
plt.ylim(0,8000)
plt.xlim(0,100)

plt.scatter(df['age'], df['creatinine_phosphokinase'])


# In[7]:


sns.countplot(x='diabetes', data=df)


# In[8]:


sns.countplot(x='smoking', data=df)


# In[9]:


sns.pairplot(df1, height=2.5)


# In[11]:


df['high_blood_pressure'].value_counts().mode()


# In[12]:


df['diabetes'].value_counts().mode()


# In[36]:


df['anaemia'].value_counts().mode()


# In[37]:


df['sex'].value_counts().mode()


# In[38]:


df['smoking'].value_counts().mode()


# In[13]:


df1 = df.sort_values(by=['age'])
df1


# In[14]:


##dataframe with age group 40-50
df2 = df1.iloc[:74]
df2


# In[15]:


##dataframe with age group 51-60
df3 = df1.iloc[75:164]
df3


# In[16]:


##dataframe with age group 61-70
df4 = df1.iloc[165:247]
df4


# In[17]:


##dataframe with age group 71-80
df5 = df1.iloc[248:281]
df5


# In[18]:


##dataframe with age group 81 and 81+
df6 = df1.iloc[282:]
df6


# In[20]:


df2['diabetes'].value_counts().mode()


# In[21]:


df3['diabetes'].value_counts().mode()


# In[22]:


df4['diabetes'].value_counts().mode()


# In[23]:


df5['diabetes'].value_counts().mode()


# In[24]:


df6['diabetes'].value_counts().mode()


# In[25]:


df2['high_blood_pressure'].value_counts().mode()


# In[26]:


df3['high_blood_pressure'].value_counts().mode()


# In[27]:


df4['high_blood_pressure'].value_counts().mode()


# In[28]:


df5['high_blood_pressure'].value_counts().mode()


# In[84]:


df6['high_blood_pressure'].value_counts().mode()


# In[29]:


def filter_diabetes_true(df):
    return df[df['diabetes'] == True]


# In[30]:


true_diabetes_patients = filter_diabetes_true(df2)

# print the filtered DataFrame
print(true_diabetes_patients)


# In[31]:


df.describe(include = 'all')


# In[32]:


df.corr()


# In[35]:


plt.figure(figsize=(14, 14))
sns.heatmap(df.corr(), annot=True)
plt.show()


# In[36]:


print(df.isnull().sum())


# In[46]:


fig = plt.figure(figsize=(20, 20))
ax = fig.gca()
df.plot(kind='density', subplots=True, layout=(4, 4), sharex=False, ax=ax)
plt.show()


# In[47]:


fig = plt.figure(figsize=(20, 20))
ax = fig.gca()
df.plot(kind='box', subplots=True, layout=(4, 4), sharex=False, ax=ax)
plt.show()


# In[49]:


bins = [40,50,60,70,80, np.nan]
labels = ['40+', '50+', '60+', '70+', '80+']

df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)

print('Percentage of people of age >=80 lose their life :', df['DEATH_EVENT'][df['age_group']=='80+'].value_counts(normalize=True)[1]*100)

sns.barplot(x='age_group', y='DEATH_EVENT', data=df)
plt.show()


# In[50]:


sns.barplot(x='anaemia', y='DEATH_EVENT', data=df)
plt.show()


# In[52]:


bins = [10, 120, np.nan]
labels = ['Normal','Abnormal']
df['creatinine_phosphokinase_group'] = pd.cut(df['creatinine_phosphokinase'], bins=bins, labels=labels)

sns.barplot(x='creatinine_phosphokinase_group', y='DEATH_EVENT', data=df)
plt.show()


# In[53]:


sns.barplot(x='diabetes', y='DEATH_EVENT', data=df)
plt.show()


# In[55]:


bins = [0, 41, 50, 70, np.nan]
labels = ['too low', 'borderline', 'Normal', 'high']
df['ejection_fraction_category'] = pd.cut(df['ejection_fraction'], bins=bins, labels=labels)

df['ejection_fraction_category'].value_counts()

sns.barplot(x='ejection_fraction_category', y='DEATH_EVENT', data=df)
plt.show()


# In[56]:


sns.barplot(x='high_blood_pressure', y='DEATH_EVENT', data=df)
plt.show()

print('Percentage of people resulted in Heart Failure having high blood pressure : ', df['DEATH_EVENT'][df['high_blood_pressure']==1].value_counts(normalize=True)[1]*100)


# In[118]:


from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import RFE

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import warnings
warnings.filterwarnings('ignore')


# In[85]:


array = df.values
X = array[:, :12]
Y = array[:, 12]


# In[86]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.67, random_state=1)


# In[87]:


scalar = StandardScaler()
rescaled_X = scalar.fit_transform(x_train)
print(rescaled_X[:5])


# In[92]:


from sklearn.feature_selection import RFE
model = LogisticRegression()
rfe = RFE(model)
fit = rfe.fit(rescaled_X, y_train)

transformed_X = fit.transform(rescaled_X)

print(df.columns)
print('Num features : ', fit.n_features_)
print('Selected features : ', fit.support_)
print('Features ranking : ', fit.ranking_)


# In[113]:


classifiers = [
    LogisticRegression(),
    DecisionTreeClassifier(),
    RandomForestClassifier()
]

for classifier in classifiers:
    scores = cross_val_score(classifier, transformed_X, y_train, cv=6)
    
    mean_accuracy = scores.mean()*100
    
    print(f"Classifier: {classifier.__class__.__name__}")
    print(f"Mean Accuracy: {mean_accuracy}")
    print("---")


# In[117]:


steps = [('scaler', StandardScaler()),
         ('RFE', RFE(LogisticRegression()))]

pipeline = Pipeline(steps)
pipeline.fit(x_train, y_train)
predictions = pipeline.predict(x_test)
print('The accuracy score of the test dataset : ', accuracy_score(y_test, predictions))
print('\nThe confusion matrix : \n', confusion_matrix(y_test, predictions))
print('\nThe classification report : \n', classification_report(y_test, predictions))
print('Score : ', pipeline.score(x_test, y_test))


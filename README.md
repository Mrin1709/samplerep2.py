1)Anaemia Prediction model
import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
dataset = pd.read_csv('C:/Users/PC/Downloads/d_output.csv')
dataset.tail(100)

dataset.info()

dataset.describe()

x = dataset.drop('Anaemic', axis=1)
x

y = dataset["Anaemic"].str.replace("Yes", '1')
y = y.str.replace("No", '0').astype(int)
y

dataset['Sex'] = dataset['Sex'].str.replace(" ", "")
dataset['Sex']

sum(dataset['Sex']=='F'), sum(dataset['Sex']=='M')

males = dataset[dataset['Sex']=='M']
females= dataset[dataset['Sex']=='F']

sum(males['Anaemic']=='Yes'), sum(males['Anaemic']=='No')

sum(females['Anaemic']=='Yes'), sum(females['Anaemic']=='No')

f={'Males Anaemic': 87,'Females Anaemic':163, 'Males Non Anaemic': 142,'Females Non Anaemic':108,}
sex = {'Males': 229, 'Females': 271}

fig, (subplot1,subplot2)   = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
subplot1.bar(f.keys(), f.values(), width=0.5)
subplot1.set(title='Sex Vs Anaemic Presence',
            xlabel='Sex',
            ylabel='Total Number')

subplot2.bar(sex.keys(), sex.values())
subplot2.set(title='Sex Vs Number',
            xlabel='Sex',
            ylabel='Number')
plt.show()

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
features_cat = ['Sex']
hot_encoder = OneHotEncoder()
transformer = ColumnTransformer([("transformer", hot_encoder, features_cat)], remainder='passthrough')
transformed_x = transformer.fit_transform(x)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(transformed_x, y, test_size=0.25)

cls = RandomForestClassifier()
cls.fit(x_train, y_train)

cls.score(x_test, y_test)

cls.predict_proba(x_test)[:5]

y_pred = cls.predict(x_test)

from sklearn.model_selection import cross_val_score
cvs = cross_val_score(cls,transformed_x,y, cv=5) 
cvs

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred) 
cm

import seaborn as sns
plt.figure(figsize=(8,6)) 
sns.heatmap(cm, annot=True)
plt.show()
2)Sexual violence
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


df_sexual_violence = pd.read_csv("C:/Users/PC/Downloads/sexual-violence-in-armed-conflict-dataset-2016-june-21-xlsx-2.csv", encoding='ISO-8859-1')
df_crime_trends = pd.read_csv("C:/Users/PC/Downloads/crime-trends-and-operations-of-criminal-justice-systems-un-cts-csv-1.csv")

df_sexual_violence.head()

df_sexual_violence.drop(columns=['Unnamed: 53'], inplace=True)

df_sexual_violence.isnull().sum()

numeric_df = df_sexual_violence.select_dtypes(include=[np.number])

plt.figure(figsize=(15, 10))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Sexual Violence in Armed Conflict Dataset')
plt.show()

df_crime_trends.head()

df_crime_trends.isnull().sum()

plt.figure(figsize=(12, 6))
sns.lineplot(data=df_crime_trends, x='date', y='sexual violence', hue='country/territory')
plt.title('Trend of Sexual Violence Over the Years')
plt.xlabel('Year')
plt.ylabel('Sexual Violence Cases')
plt.legend(loc='upper right')
plt.show()

df_crime_trends['target'] = df_crime_trends['sexual violence'].apply(lambda x: 1 if x > 0 else 0)
X = df_crime_trends[['date', 'rate']]
y = df_crime_trends['target']

X.fillna(X.mean(), inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

accuracy, conf_matrix, class_report
3)UPI transactions
import numpy as np 
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
df = pd.read_csv("C:/Users/PC/Downloads/transactions.csv")

df.info()
df.head()

print(df.describe())
print(df.isnull().sum())

print(df.columns)

with open('/kaggle/input/upi-payment-transactions-dataset/transactions.csv', 'r') as file:
    for i in range(5):  # Print the first 5 lines
        print(file.readline().strip())
        
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("C:/Users/PC/Downloads/transactions.csv")

label_encoder = LabelEncoder()
df['Sender Name'] = label_encoder.fit_transform(df['Sender Name'])
df['Sender UPI ID'] = label_encoder.fit_transform(df['Sender UPI ID'])
df['Receiver Name'] = label_encoder.fit_transform(df['Receiver Name'])
df['Receiver UPI ID'] = label_encoder.fit_transform(df['Receiver UPI ID'])
df['Status'] = label_encoder.fit_transform(df['Status'])  

if 'Timestamp' in df.columns:
   
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Year'] = df['Timestamp'].dt.year
    df['Month'] = df['Timestamp'].dt.month
    df['Day'] = df['Timestamp'].dt.day
    df['Hour'] = df['Timestamp'].dt.hour

   
    df.drop(columns=['Timestamp'], inplace=True)


X = df.drop(columns=['Status'])
y = df['Status']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

categorical_cols = ['Sender Name', 'Sender UPI ID', 'Receiver Name', 'Receiver UPI ID']

encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

X_train_encoded = X_train.copy()
X_train_encoded = pd.DataFrame(encoder.fit_transform(X_train[categorical_cols]), 
                               columns=encoder.get_feature_names_out(categorical_cols),
                               index=X_train.index)

X_test_encoded = X_test.copy()
X_test_encoded = pd.DataFrame(encoder.transform(X_test[categorical_cols]), 
                              columns=encoder.get_feature_names_out(categorical_cols),
                              index=X_test.index)


xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train_encoded, y_train)

y_pred_xgb = xgb_model.predict(X_test_encoded)

print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("XGBoost Classification Report:\n", classification_report(y_test, y_pred_xgb))
print("XGBoost Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))    
4)ICC T20 WC
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

fielding_stats = pd.read_csv("C:/Users/PC/Downloads/fielding_stats_for_icc_mens_t20_world_cup_2024.csv")
wk_stats = pd.read_csv("C:/Users/PC/Downloads//wk_stats_for_icc_mens_t20_world_cup_2024.csv")
batting_stats = pd.read_csv("C:/Users/PC/Downloads//batting_stats_for_icc_mens_t20_world_cup_2024.csv")
match_results = pd.read_csv("C:/Users/PC/Downloads/match_results_for_icc_mens_t20_world_cup_2024.csv")
bowling_stats = pd.read_csv("C:/Users/PC/Downloads/bowling_stats_for_icc_mens_t20_world_cup_2024.csv")

display(fielding_stats.head())
display(wk_stats.head())
display(batting_stats.head())
display(match_results.head())
display(bowling_stats.head())

match_results['Match Date'] = pd.to_datetime(match_results['Match Date'])

plt.figure(figsize=(12, 6))
sns.countplot(data=match_results, x='Winner', order=match_results['Winner'].value_counts().index)
plt.xticks(rotation=90)
plt.title('Number of Matches Won by Each Team')
plt.xlabel('Team')
plt.ylabel('Number of Wins')
plt.show()

numeric_df = batting_stats.select_dtypes(include=[np.number])
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap for Batting Stats')
plt.show()

match_results['Team 1'] = match_results['Team 1'].astype('category').cat.codes
match_results['Team 2'] = match_results['Team 2'].astype('category').cat.codes
match_results['Winner'] = match_results['Winner'].astype('category').cat.codes

X = match_results[['Team 1', 'Team 2']]
y = match_results['Winner']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

accuracy, conf_matrix, class_report
5)Food Nutrition
import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
dfs = []
for i in range(1, 6):
    df=pd.read_csv("C:/Users/PC/Downloads/Food-Nutrition.csv")
    
df = pd.concat(dfs, ignore_index=True)
df.info()
df.isnull().sum()
df.head()

df.describe()

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(df.iloc[:,0:13],df.iloc[:,-1],test_size=0.2,random_state=21)

from sklearn.preprocessing import MinMaxScaler

scalar = MinMaxScaler()

X_train_transformed = scalar.fit_transform(X_train)
X_test_transformed = scalar.transform(X_test)

X_train_transformed.shape,X_test_transformed.shape

from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor()
model.fit(X_train_transformed,y_train)
y_pred = model.predict(X_test_transformed)

from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_pred,y_test)
print(mae)

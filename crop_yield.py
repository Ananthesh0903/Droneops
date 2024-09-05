import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load and preprocess the data
df = pd.read_csv('yield.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)
df['average_rain_fall_mm_per_year'] = df['average_rain_fall_mm_per_year'].astype(np.float64)

# Remove invalid entries
def isStr(obj):
    try:
        float(obj)
        return False
    except:
        return True

to_drop = df[df['average_rain_fall_mm_per_year'].apply(isStr)].index
df = df.drop(to_drop)

# Data visualization
plt.figure(figsize=(10, 20))
sns.countplot(y=df['Area'])
district = df['Area'].unique()
yield_per_district = [df[df['Area'] == state]['hg/ha_yield'].sum() for state in district]
plt.figure(figsize=(15, 20))
sns.barplot(y=district, x=yield_per_district)

sns.countplot(y=df['Item'])
fruits = df['Item'].unique()
yield_per_fruit = [df[df['Item'] == fruit]['hg/ha_yield'].sum() for fruit in fruits]
sns.barplot(y=fruits, x=yield_per_fruit)

# Prepare data for modeling
col = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'average_temp', 'Area', 'Item', 'hg/ha_yield']
df = df[col]
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0, shuffle=True)

# Preprocessing
ohe = OneHotEncoder(drop='first')
scale = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('StandardScale', scale, [0, 1, 2, 3]),
        ('OHE', ohe, [4, 5]),
    ],
    remainder='passthrough'
)

X_train_dummy = preprocessor.fit_transform(X_train)
X_test_dummy = preprocessor.transform(X_test)

# Train models
models = {
    'lr': LinearRegression(),
    'lss': Lasso(),
    'Rid': Ridge(),
    'Dtr': DecisionTreeRegressor()
}

for name, md in models.items():
    md.fit(X_train_dummy, y_train)
    y_pred = md.predict(X_test_dummy)
    print(f"{name} : mae : {mean_absolute_error(y_test, y_pred)} score : {r2_score(y_test, y_pred)}")

dtr = DecisionTreeRegressor()
dtr.fit(X_train_dummy, y_train)
dtr.predict(X_test_dummy)

def prediction(Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item):
    features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]], dtype=object)
    transformed_features = preprocessor.transform(features)
    predicted_yield = dtr.predict(transformed_features).reshape(1, -1)
    return predicted_yield[0]

Year = 1990
average_rain_fall_mm_per_year = 1485.0
pesticides_tonnes = 121.00
avg_temp = 16.37
Area = 'Shimla'
Item = 'Fengal'
result = prediction(Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item)
print(result)

# Save models and preprocessor with joblib
joblib.dump(dtr, 'dtr.joblib')
joblib.dump(preprocessor, 'preprocessor.joblib')

# Load models and preprocessor
dtr = joblib.load('dtr.joblib')
preprocessor = joblib.load('preprocessor.joblib')

# Verify scikit-learn version
import sklearn
print(sklearn.__version__)

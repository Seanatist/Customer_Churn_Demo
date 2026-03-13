import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

df = pd.read_csv("telco_customer.csv")

df.shape

df.head()

pd.set_option("display.max_columns", None)

df.info()

df = df.drop(columns = ["customerID"])

df.head()

df.columns

print(df["gender"].unique())

print(df["SeniorCitizen"].unique())

numerical_features_list = ["tenure", "MonthlyCharges", "TotalCharges"]

for col in df.columns:
  if col not in numerical_features_list:
     print(col, df[col].unique())
     print("-"*50)



df.isnull().sum()

df[df["TotalCharges"] == " "]

len(df[df["TotalCharges"] == " "])

df["TotalCharges"] = df["TotalCharges"].replace({" " : 0.0})

df["TotalCharges"] = df["TotalCharges"].astype(float)

df.info()

print(df["Churn"].value_counts())

df.shape

df.describe()

from matplotlib import lines
def plot_histogram(df, column_name):
    plt.figure(figsize=(5,3))
    sns.histplot(df[column_name], kde=True)
    plt.title(f"Distribution of {column_name}")
    plt.show()

    col_mean = df[column_name].mean()
    col_median = df[column_name].median()


    plt.axvline(col_mean, color = "red", linestyle = " ", label = "Mean")
    plt.axvline(col_median, color = "green", linestyle = " ", label = "Median")
    plt.legend()
    plt.show()


plot_histogram(df, "tenure")

plot_histogram(df, "MonthlyCharges")

plot_histogram(df, "TotalCharges")

def plot_box_plot(df, column_name):

  plt.figure(figsize=(5,3))
  sns.boxplot(y = df[column_name])
  plt.title(f"Box plot of {column_name}")
  plt.show()

plot_box_plot(df, "tenure")

plot_box_plot(df, "MonthlyCharges")

plot_box_plot(df, "TotalCharges")

plt.figure(figsize=(8,4))
sns.heatmap(df[["tenure", 'MonthlyCharges', "TotalCharges"]].corr(), annot=True,cmap ="coolwarm", fmt = '2f' )
plt.title("Correlation Heatmap")
plt.show()

df.columns

df.info()

object_cols = df.select_dtypes(include = "object").columns.to_list()

object_cols = ["SeniorCitizen"] + object_cols

for col in object_cols:
  plt.figure(figsize=(5,3))
  sns.countplot(x = df[col])
  plt.title(f"Count Plot of {col}")
  plt.show()

df["Churn"] = df["Churn"].replace({"No" : 0,})

df["Churn"].value_counts()

object_columns = df.select_dtypes(include = "object").columns

print(object_columns)



encoders = {}

for column in object_columns:
    df[column] = df[column].astype(str)   # 🔥 Force all values to string

    label_encoder = LabelEncoder()
    df[column] = label_encoder.fit_transform(df[column])
    encoders[column] = label_encoder

with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)


encoders

df.head()

x = df.drop(columns=["Churn"])
y = df["Churn"]

pass

df.head()

print(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print(y_train.shape)

print(y_train.value_counts())

smote = SMOTE(random_state=42)
x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)


print(y_train_smote.shape)

print(y_train_smote.value_counts())

models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier()
}

cv_scores = {}
for model_name, model in models.items():

  print(f"Training {model_name} with default parameters")
  scores = cross_val_score(model, x_train_smote, y_train_smote, cv=5, scoring = "accuracy")
  cv_scores[model_name] = scores
  print(f"{model_name} cross_validation accuracy : (no_mean{scores}:2f")
  print("-"*70)


cv_scores

rfc = RandomForestClassifier(random_state= 42)
rfc.fit(x_train_smote, y_train_smote)

print(y_test.value_counts())

model = XGBClassifier()
model.fit(x_train, y_train)

# 2. Predict
y_test_pred = model.predict(x_test)

y_test_pred = model.predict(x_test)

print("Accuracy Score:\n, accuracy_score(y_test, y_test_pred)")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
print("Classification Report:\n", classification_report(y_test, y_test_pred))

model_data = {"model" : rfc, "features_names": x.columns.tolist()}

with open("customer_churn_model.pkl", "wb") as f:
    pickle.dump(model_data, f)

with open("customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)

loaded_model = model_data["model"]
feature_names = model_data["features_names"]

print(loaded_model)

print(feature_names)

import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Input data
input_data = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85
}

input_df = pd.DataFrame([input_data])

# Load saved encoders
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# Transform columns safely
for col, encoder in encoders.items():
    if col in input_df.columns:  # Check if column exists in input_df
        input_df[col] = input_df[col].astype(str)

        # Handle unseen labels by adding a new category temporarily
        classes = list(encoder.classes_)
        new_values = [x for x in input_df[col] if x not in classes]
        if new_values:
            # Extend encoder classes
            encoder.classes_ = np.append(encoder.classes_, new_values)

        # Transform
        input_df[col] = encoder.transform(input_df[col])

# Load model
with open("customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)

loaded_model = model_data["model"]

# Make prediction
prediction = loaded_model.predict(input_df)
pred_prob = loaded_model.predict_proba(input_df)

print(f"Prediction: {'churn' if prediction[0]==1 else 'No churn'}")
print(f"Prediction Probability: {pred_prob}")

encoders



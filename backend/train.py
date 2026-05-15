import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

df = pd.read_csv("../data/clean_dataset.csv")

X = df.drop("risk", axis=1)
y = df["risk"]

le = LabelEncoder()
y = le.fit_transform(y)

model = XGBClassifier()
model.fit(X, y)

# save model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(le, open("encoder.pkl", "wb"))
pickle.dump(X.columns.tolist(), open("features.pkl", "wb"))

print("✅ Model trained and saved")
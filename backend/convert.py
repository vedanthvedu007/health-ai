import pandas as pd

df = pd.read_csv("../data/dataset.csv")  # your Kaggle file

# collect all symptoms
all_symptoms = set()

for col in df.columns:
    if "Symptom" in col:
        df[col] = df[col].fillna("")
        for val in df[col]:
            if val != "":
                all_symptoms.add(val.strip())

all_symptoms = list(all_symptoms)

# create columns
for symptom in all_symptoms:
    df[symptom] = 0

# fill 0/1
for i, row in df.iterrows():
    for col in df.columns:
        if "Symptom" in col:
            sym = row[col]
            if sym in all_symptoms:
                df.at[i, sym] = 1

# risk mapping
def map_risk(disease):
    d = str(disease).lower()
    if "heart" in d or "attack" in d or "stroke" in d:
        return "HIGH"
    elif "infection" in d or "fever" in d:
        return "MEDIUM"
    else:
        return "LOW"

df["risk"] = df["Disease"].apply(map_risk)

# drop old columns
drop_cols = [c for c in df.columns if "Symptom" in c or c == "Disease"]
df = df.drop(columns=drop_cols)

df.to_csv("clean_dataset.csv", index=False)

print("✅ Dataset ready: clean_dataset.csv")
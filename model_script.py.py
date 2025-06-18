#!/usr/bin/env python
# coding: utf-8

# In[5]:


print("Train columns:\n", train.columns.tolist())
print("Test columns:\n", test.columns.tolist())


# In[7]:


print("Medical history columns:\n", med.columns.tolist())


# In[9]:


print(med.columns)


# In[37]:


import pandas as pd

# Load datasets
train = pd.read_csv("train.csv")
test = pd.read_csv("test_share.csv")
med = pd.read_csv("medical_history.csv")
dem = pd.read_csv("demographic_details.csv")

# Merge with medical and demographic data
train = train.merge(med, on="PatientId", how="left")
train = train.merge(dem, on="PatientId", how="left")

test = test.merge(med, on="PatientId", how="left")
test = test.merge(dem, on="PatientId", how="left")


# In[39]:


def process(df):
    df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"])
    df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"])
    df["WaitingDays"] = (df["AppointmentDay"] - df["ScheduledDay"]).dt.days
    df["DayOfWeek"] = df["AppointmentDay"].dt.dayofweek
    df["IsWeekend"] = df["DayOfWeek"] >= 5
    df["AgeGroup"] = pd.cut(df["Age"], bins=[0, 18, 40, 60, 120], labels=["Child", "Adult", "Middle", "Elderly"])
    df["HasChronic"] = ((df["Hipertension"] == 1) | (df["Diabetes"] == 1)).astype(int)
    df["ScheduledHour"] = df["ScheduledDay"].dt.hour
    df["AppointmentMonth"] = df["AppointmentDay"].dt.month
    return df

train = process(train)
test = process(test)


# In[41]:


from sklearn.preprocessing import LabelEncoder

cat_cols = ["Gender", "Neighbourhood", "AgeGroup"]
le = LabelEncoder()
for col in cat_cols:
    train[col] = le.fit_transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))


# In[43]:


X = train.drop(["No-show", "PatientId", "AppointmentID", "ScheduledDay", "AppointmentDay"], axis=1)
y = train["No-show"].map({"Yes": 1, "No": 0})

X_test = test.drop(["PatientId", "AppointmentID", "ScheduledDay", "AppointmentDay"], axis=1)


# In[45]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

model = RandomForestClassifier(n_estimators=100, random_state=42)
scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")

print(f"ROC AUC CV Score: {scores.mean():.4f}")


# In[46]:


model.fit(X, y)
pred_probs = model.predict_proba(X_test)[:, 1]  # predicted probabilities for "No-show"


# In[66]:


print("Min prob:", pred_probs.min())
print("Max prob:", pred_probs.max())
print("Unique extremes:", np.unique(pred_probs[(pred_probs == 0) | (pred_probs == 1)]))


# Predict on test set
pred_probs = model.predict_proba(X_test)[:, 1]

# Check predicted probabilities before saving
print(pred_probs[:10])               # Sample predictions
print(pred_probs.min(), pred_probs.max())  # Range of probabilities
print(np.unique(pred_probs))        # Check for variety (not all 0 or 1)

# Prepare submission
submission = pd.DataFrame({
    "PatientId": test["PatientId"],
    "No-show": pred_probs
})


submission.to_csv("Md_Al_Emran_Attempt1.csv", index=False)


# In[67]:


print("Test rows:", test.shape[0])
print("Submission rows:", submission.shape[0])


# In[ ]:





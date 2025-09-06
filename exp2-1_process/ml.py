import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pickle
import collections

folder_path = 'C:\\codehome\\eegdata\\output'

df_n = pd.read_csv(os.path.join(folder_path, '1-1_power_n.csv'))
df_n['label'] = 1

df_p = pd.read_csv(os.path.join(folder_path, '1-1_power_p.csv'))
df_p['label'] = 0

df_all = pd.concat([df_n, df_p], ignore_index=True)

groups_per_sample = 36
df_all['sample_id'] = df_all.index // groups_per_sample

print(df_all)

df_all['band_region'] = df_all['band'] + '_' + df_all['region']

df_pivot = df_all.pivot(index='sample_id', columns='band_region', values='power_change')


print(df_pivot)
labels = df_all.groupby('sample_id')['label'].first()
df_pivot['label'] = labels

df_pivot = df_pivot.fillna(0)

X = df_pivot.drop('label', axis=1)
y = df_pivot['label'].values

print("Label distribution:")
print(collections.Counter(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

xgb = XGBClassifier(eval_metric='logloss', random_state=42)
xgb.fit(X_train_scaled, y_train)

y_pred = xgb.predict(X_test_scaled)
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred))

feature_importances = xgb.feature_importances_
for name, score in sorted(zip(X.columns, feature_importances), key=lambda x: x[1], reverse=True):
    print(f"{name}: {score:.4f}")

from sklearn.svm import SVC
from sklearn.metrics import classification_report


svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train_scaled, y_train)

y_pred_svm = svm.predict(X_test_scaled)
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

with open('1-1scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('1-1imputer.pkl', 'wb') as f:
    pickle.dump(imputer, f)
with open('1-1xgb_model.pkl', 'wb') as f:
    pickle.dump(xgb, f)
with open('1-1feature_columns.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

pickle.dump(svm, open('1-1svm_model.pkl', 'wb'))

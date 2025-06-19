# ✅ STEP 1: Install & Import Required Libraries
# (No extra installation needed for this basic ML project)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# ✅ STEP 2: Load Dataset from UCI Repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv"
df = pd.read_csv(url)

# ✅ STEP 3: Display Data Info
print("🔥 Dataset Preview:")
print(df.head())
print("\n✅ Dataset Info:")
print(df.info())
print("\n📊 Statistical Summary:")
print(df.describe())

# ✅ STEP 4: Encode Categorical Features (month, day)
le_month = LabelEncoder()
le_day = LabelEncoder()
df['month'] = le_month.fit_transform(df['month'])
df['day'] = le_day.fit_transform(df['day'])

# ✅ STEP 5: Create Target Variable (Fire: 0 or 1)
df['fire'] = (df['area'] > 0).astype(int)  # 1 = fire occurred, 0 = no fire
df.drop('area', axis=1, inplace=True)     # Drop area as we do classification

# ✅ STEP 6: Split Features and Labels
X = df.drop('fire', axis=1)
y = df['fire']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ✅ STEP 7: Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ✅ STEP 8: Predict and Evaluate
y_pred = model.predict(X_test)

print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))
print("🔍 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ✅ STEP 9: Plot Feature Importance
feature_importance = pd.Series(model.feature_importances_, index=X.columns)
feature_importance.sort_values(ascending=True).plot(kind='barh', figsize=(10, 6), title='Feature Importance')
plt.xlabel("Importance Score")
plt.show()

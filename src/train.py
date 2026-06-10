import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import pickle

# Load features
df = pd.read_csv('../data/features.csv')

# Define input features (X) and target (y)
X = df[['home_win_rate', 'away_win_rate',
        'home_draw_rate', 'away_draw_rate',
        'home_goals_scored', 'away_goals_scored',
        'home_goals_conceded', 'away_goals_conceded',
        'home_form', 'away_form']]

y = df['result']
X = X.fillna(0)
# Split data - 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# --- MODEL 1: Logistic Regression (baseline) ---
print("\n--- Logistic Regression ---")
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, lr_preds), 3))
print(classification_report(y_test, lr_preds))

# --- MODEL 2: XGBoost (main model) ---
print("\n--- XGBoost ---")
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    random_state=42
)
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, xgb_preds), 3))
print(classification_report(y_test, xgb_preds))

# Save the best model
with open('../src/model.pkl', 'wb') as f:
    pickle.dump(xgb, f)

print("\nModel saved as model.pkl")
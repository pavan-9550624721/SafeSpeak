import joblib
from sklearn.metrics import accuracy_score, f1_score
from feature_engineering import run_feature_engineering

print("Loading data...")
X_train_sparse, X_val_sparse, y_train, y_val, feature_names = run_feature_engineering(
    "train.csv"
)

print("Evaluating XGBoost...")
xgb_model = joblib.load("model/xgboost_model.pkl")
y_pred_prob_xgb = xgb_model.predict_proba(X_val_sparse)
y_pred_bin_xgb = (y_pred_prob_xgb >= 0.5).astype(int)

print("Evaluating Random Forest...")
rf_model = joblib.load("model/random_forest_model.pkl")
y_pred_prob_rf = rf_model.predict_proba(X_val_sparse)
y_pred_bin_rf = (y_pred_prob_rf >= 0.5).astype(int)

print("\n" + "=" * 50)
print("XGBOOST VS RANDOM FOREST COMPARISON SUMMARY")
print("=" * 50)
print(f"XGBoost Accuracy:      {accuracy_score(y_val, y_pred_bin_xgb):.4f}")
print(f"Random Forest Accuracy:{accuracy_score(y_val, y_pred_bin_rf):.4f}")
print(f"XGBoost Micro F1:      {f1_score(y_val, y_pred_bin_xgb, average='micro'):.4f}")
print(f"Random Forest Micro F1:{f1_score(y_val, y_pred_bin_rf, average='micro'):.4f}")
print("=" * 50)

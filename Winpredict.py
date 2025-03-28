import pandas as pd  # Data handling
import numpy as np  # Numerical computations

# XGBoost Model & Utilities
import xgboost as xgb  
from xgboost import XGBClassifier  

# Sklearn Utilities
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler  
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss



df = pd.read_csv("/Users/arunramji/Desktop/Waterloo Wins Corr/Team Stats(Sheet1) (2).csv")
df["Points"]= df["Result"].apply(lambda x: 2 if x == "W" else 1 if x == "T" else 0)

X = df.drop(columns=["Points","XG","NF+","NF-","Odd+","Odd-","XGF","XGA","Game","Game ID","Even Strength Time ","GFMod","GAMod","Result","A+","A-","AD",
                     "CE","UE","Rush","Fore","Cycle","Faceoff","GF","GA"])
y = df['Points']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



#Options for fine tuning 
param_grid = {
  'n_estimators':np.arange(100,400,50),
   'learning_rate': np.linspace(0.01,0.2,5),
   'max_depth': np.arange(3,8,1),
   'subsample':np.linspace(0.7,1,4),
   'colsample_bytree': np.linspace(0.7,1,4)

}

# Initialize model 
model = xgb.XGBClassifier(
    objective='multi:softprob',
    eval_metric = "mlogloss",
    random_state = 42

    
)

# Find best values for tuning 
grid_search = GridSearchCV(
    estimator = model,
    param_grid = param_grid,
    scoring = "neg_log_loss",
    cv = 5,
    verbose=1,
    n_jobs = -1

)

grid_search.fit(X_train,y_train)

# Get predicted probabilities for Points (0,1,2)
y_prob = grid_search.best_estimator_.predict_proba(X_test)

# Compute Log Loss
logloss = log_loss(y_test, y_prob)
print("Final Log Loss Score:", logloss)

#Feature Importance
best_model = grid_search.best_estimator_
feature_names = X_train.columns
importance_scores = best_model.feature_importances_
importance_df = pd.DataFrame({'Feature':feature_names,
'Importance':importance_scores})

# Sort features by importance (highest first)
importance_df = importance_df.sort_values(by="Importance", ascending=False)

# Display top 20 features
print(importance_df.head(20))
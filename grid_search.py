
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

df = pd.read_csv('data/smote_data.csv')

X = df.drop(['status_group'], axis=1)
y = df['status_group']

X_train, X_test, y_train, y_test = train_test_split(X,y, stratify=y, random_state=42)

numeric_features = X_train.select_dtypes(['int64', 'float64']).columns.to_list()
cat_features = X_train.select_dtypes(['object', 'bool']).columns.to_list()

numeric_transformer = Pipeline(steps=[('scaler',  StandardScaler())])

cat_transformer = OneHotEncoder(drop='first', handle_unknown='infrequent_if_exist')

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', cat_transformer, cat_features)])

clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier(random_state=42))])

param_grid = {
    'classifier__n_estimators': range(0,5000,100),
    'classifier__max_depth': [None, 500, 1000, 1500, 2500, 5000],
    'classifier__min_samples_split': range(1,50,2),
    'classifier__min_samples_leaf': range(1,50,2),
    'classifier__max_features': ['sqrt','log2', None],
    'classifier__bootstrap': [True, False],
    'classifier__oob_score': [True, False],
    'classifier__ccp_alpha': np.linspace(0,1,9),
    # 'classifier__class_weight': ['balanced', 'balanced_subsample', None]
}

grid_search_rf = GridSearchCV(clf,param_grid, n_jobs=8, scoring='accuracy')

grid_search_rf.fit(X_train, y_train)

y_hat = grid_search_rf.predict(X_test)

pickle_out = open('models/rf_grid_search', 'wb')
pickle.dump(grid_search_rf, pickle_out)
pickle_out.close()

print('Accuracy Score:', accuracy_score(y_test, y_hat))

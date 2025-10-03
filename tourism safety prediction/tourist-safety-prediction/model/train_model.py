import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
import pickle

# Load dataset
df = pd.read_csv('data/combined_tourism_dataset_labeled.csv')

# Features and target
X = df.drop(['score', 'label', 'Preferred_Place', 'Tourist_Type_Prevalence'], axis=1)
y = df['score']

# Define categorical columns
cat_columns = X.columns.tolist()

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OrdinalEncoder(), cat_columns)
    ])

# Model pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train model
pipeline.fit(X, y)

# Save model and preprocessor
with open('model/tourism_model.pkl', 'wb') as f:
    pickle.dump(pipeline.named_steps['model'], f)
with open('model/preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)
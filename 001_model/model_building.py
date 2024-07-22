import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


df = pd.read_csv("haberman.csv")
scaler = StandardScaler()

# Define features and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1] - 1 # Convert survival_status to binary (0 for survival, 1 for death within 5 years)

# Standardize features
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Sample dataset
data = {
    'Age': [25, 32, 28, 45, 30, 35, 40, 50, 22, 27],
    'BMI': [22.5, 28.3, 24.0, 30.5, 26.4, 23.0, 27.8, 29.9, 21.0, 25.2],
    'ExerciseFreq': [4, 1, 3, 0, 2, 4, 1, 0, 5, 3],
    'HeartRate': [65, 85, 70, 90, 78, 66, 82, 88, 60, 72],
    'FitnessStatus': ['Fit', 'Not Fit', 'Fit', 'Not Fit', 'Fit',
                      'Fit', 'Not Fit', 'Not Fit', 'Fit', 'Fit']
}

# Create a DataFrame
df = pd.DataFrame(data)

# Encode labels
le = LabelEncoder()
df['FitnessEncoded'] = le.fit_transform(df['FitnessStatus'])  # Fit = 0, Not Fit = 1 or vice versa

# Split features and labels
X = df[['Age', 'BMI', 'ExerciseFreq', 'HeartRate']]
y = df['FitnessEncoded']

# Train the model
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Save the model and label encoder
with open("fitness_model.pkl", "wb") as f:
    pickle.dump((model, le), f)

print("✅ fitness_model.pkl file has been created successfully!")

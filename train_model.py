import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# Load dataset
dataset = pd.read_csv("data/Training.csv")

X = dataset.drop("prognosis", axis=1)
y = dataset["prognosis"]

le = LabelEncoder()
Y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=20
)

# Train model
svc = SVC(kernel="linear")
svc.fit(X_train, y_train)

# Save model
pickle.dump(svc, open("model/svc.pkl", "wb"))

print("✅ Model trained and saved successfully")

import joblib
model = joblib.load("ieee13/classification/models/KNN.pkl")
print(model.classes_)
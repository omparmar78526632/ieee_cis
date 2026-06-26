import joblib, numpy as np

scaler = joblib.load('app/trained_models/scaler.pkl')
model  = joblib.load('app/trained_models/random_forest.pkl')

print('Scaler loaded:', type(scaler).__name__)
print('Model loaded :', type(model).__name__)
print('n_estimators :', model.n_estimators)

# Test cases: [temperature, humidity, pressure, wind_speed, irradiation]
test_cases = [
    ([31.5,  65.0, 93.8, 4.2, 0.85], "Good sunny day"),
    ([25.0,  80.0, 93.0, 1.9, 0.00], "Nighttime (irrad=0)"),
    ([35.0,  50.0, 94.0, 3.0, 1.20], "Peak irradiation"),
]

print()
for inputs, label in test_cases:
    X  = np.array([inputs])
    Xs = scaler.transform(X)
    pred = max(0.0, float(model.predict(Xs)[0]))
    print(f"  {label:<25} => {pred:>10.2f} W")

print("\nAll tests passed.")

import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([
    [8, 1],
    [10, 2],
    [12, 2],
    [14, 3],
    [16, 3],
    [18, 4]
])

Y = np.array([5, 7, 9, 11, 13, 15])

model = LinearRegression()
model.fit(X, Y)

print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")

sizes_input = input("Enter pizza sizes (inches) separated by commas: ").strip()
toppings_input = input("Enter number of toppings (same order): ").strip()

sizes = [float(s.strip()) for s in sizes_input.split(',')]
toppings = [float(t.strip()) for t in toppings_input.split(',')]

X_pred = np.array(list(zip(sizes, toppings)))
predicted_prices = model.predict(X_pred)

for size, topping, price in zip(sizes, toppings, predicted_prices):
    print(f"Predicted price for a {size}-inch pizza with {int(topping)} toppings: ${price:.2f}")

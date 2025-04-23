import numpy as np
import matplotlib.pyplot as plt

X = np.array([8, 10, 12, 14, 16, 18])
Y = np.array([5, 7, 9, 11, 13, 15])

N = len(X)
m = (N * np.sum(X * Y) -np.sum(X) * np.sum(Y))/ (N * np.sum(X**2) - (np.sum(X))**2)
b = (np.sum(Y) - m * np.sum(X)) / N

print(f"Slope(m): {m}")
print(f"Intercept (b): {b}")

Y_pred = m * X + b

plt.scatter(X, Y, color='yellow', label='Data Points')
plt.plot(X, Y_pred, color = 'blue', label = 'Regression Line')
plt.xlabel('Pizza Size (inches)')
plt.ylabel('Pizza Price (dollars)')
plt.title('Pizza Price Prediction using Linear Regression')
plt.legend()
plt.show()

def predict_price(size):
	return m * size + b

sizes_to_predict = input("Enter pizza sizes (in inches) separated by commas:")
sizes_to_predict = [float(size.strip()) for size in sizes_to_predict.split(',')]

for size in sizes_to_predict:
	predicted_price = predict_price(size)
	print(f"Predicted price for a {size} - inch pizza: ${predicted_price:.2f}")
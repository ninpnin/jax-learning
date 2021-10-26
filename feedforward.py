from jax import grad, vmap
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt

def relu(x):
	return jnp.maximum(x, 0.)

def layer(theta_l, x, activation=True):
	W, b = theta_l
	x = W @ x + b.reshape(b.shape[0],1)
	if activation:
		return relu(x)
	else:
		return x

def model(theta, x0):
	x = x0
	for theta_l in theta[:-1]:
		x = layer(theta_l, x)
	x = layer(theta[-1], x, activation=False)
	return x

def loss(theta, x, y):
	y_hat = model(theta, x)
	residuals = y_hat - y
	return jnp.mean(residuals * residuals)

def plot(theta):
	x = np.array([np.linspace(0., np.pi * 2.)])
	y_hat = model(theta, x)
	x, y_hat = x[0], y_hat[0]
	plt.plot(x, y_hat)

	y = np.sin(x)
	plt.plot(x, y)
	plt.show()

def main():
	in_width = 1
	out_width = 1

	# Create data from the sine function
	x = np.random.rand(in_width, 1500) * 2 * np.pi
	vsin = np.vectorize(np.math.sin)
	y = vsin(x)

	layers = 3
	theta = []
	widths = [in_width, 25, 25, 25, out_width]
	for h_out, h_in in zip(widths[1:], widths[:-1]):
		W_l = np.random.normal(size=(h_out,h_in)) * 0.1
		b_l = np.random.normal(size=(h_out)) * 0.1
		theta.append((W_l, b_l))

	# Perform gradient descent
	lr = 0.1
	loss_prime = grad(loss)
	for i in range(5000):
		l = loss(theta, x, y)
		gradient = loss_prime(theta, x, y)
		new_theta = []
		for theta_l, dtheta_l in zip(theta, gradient):
			W, b = theta_l
			dW, db = dtheta_l

			W, b = W - lr * dW, b - lr * db 

			new_theta.append((W,b))

		theta = new_theta
		print(l)
	
	# Plot against the sine curve
	plot(theta)

if __name__ == '__main__':
	main()
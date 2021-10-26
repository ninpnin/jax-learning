from jax import grad, vmap
import jax.numpy as jnp
import numpy as np

def relu(x):
	return jnp.maximum(x, 0.)

def layer(theta_l, x):
	W = theta_l[0]
	b = theta_l[1]
	x = W @ x + b
	print("X", x.shape)
	return relu(x)

def model(theta, x0):
	x = x0
	for theta_l in theta:
		x = layer(theta_l, x)
	return x

def loss(theta, x, y):
	y_hat = model(theta, x)
	residuals = y_hat - y
	return jnp.sum(residuals * residuals)

def main():
	loss_prime = grad(loss)
	x = np.random.rand(2)
	y = np.random.rand(2)

	layers = 3
	theta = []
	for l in range(layers):
		W_l = np.random.normal(size=(2,2))
		b_l = np.random.normal(size=(2))
		theta.append((W_l, b_l))

	print("x", x)
	print("layer(x)", model(theta, x))
	print("layer'(x)", loss(theta, x, y))
	print("layer'(x)", loss_prime(theta, x, y))

if __name__ == '__main__':
	main()
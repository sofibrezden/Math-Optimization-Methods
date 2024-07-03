import numpy as np
from scipy.optimize import minimize


class FibonacciSearch:
    def search(self, f, a_init, b_init, epsilon):
        vars = round((b_init - a_init) / epsilon)
        n = self.get_iteration(vars) + 1
        x1 = a_init + (self.fib(n - 2) / self.fib(n)) * (b_init - a_init)
        x2 = a_init + (self.fib(n - 1) / self.fib(n)) * (b_init - a_init)
        fx1 = f(x1)
        fx2 = f(x2)
        k = 0
        iterations = 0
        while abs(b_init - a_init) >= epsilon:
            iterations += 1
            if fx1 <= fx2:
                b_init, x2 = x2, x1
                fx2 = fx1
                x1 = a_init + (self.fib(n - k - 3) / self.fib(n - k - 1)) * (b_init - a_init)
                fx1 = f(x1)
            else:
                a_init, x1 = x1, x2
                fx1 = fx2
                x2 = a_init + (self.fib(n - k - 2) / self.fib(n - k - 1)) * (b_init - a_init)
                fx2 = f(x2)
            k += 1
        return (a_init + b_init) / 2, iterations

    def fib(self, n):
        i, j, c = 1, 1, 0
        while c < n:
            i, j = j, i + j
            c += 1
        return i

    def get_iteration(self, vars):
        i, j, c = 1, 1, 0
        while i <= vars:
            i, j = j, i + j
            c += 1
        return c


class GradientDescentWithFibonacci:
    def __init__(self, f, dydx1, dydx2, epsilon, x_start=np.array([0, 0])):
        self.f = f
        self.dydx1 = dydx1
        self.dydx2 = dydx2
        self.epsilon = epsilon
        self.x_start = x_start

    def solve(self):
        x_curr = self.x_start
        x_prev = x_curr + 1000 * self.epsilon

        itr = 0
        while np.abs(self.f(x_prev) - self.f(x_curr)) > self.epsilon:
            itr += 1
            x_prev = x_curr
            # Цей крок визначає, наскільки далеко ми
            # будемо рухатися в кожному кроці градієнтного спуску.
            # beta для керування величиною кроку у відповідний напрям.
            x_curr = x_prev - self.get_beta(x_prev) * self.get_gradient(x_prev)

            print(
                f"Iteration {itr} | F = {self.f(x_curr)} | X = {x_curr} | grad = {self.get_gradient(x_curr)}")

        return self.f(x_curr), x_curr

    def get_gradient(self, x):
        return np.array([self.dydx1(x), self.dydx2(x)])

    def get_beta(self, x):
        fib = FibonacciSearch()

        def approximation(beta):
            return self.f(x - beta * self.get_gradient(x))

        beta, *other = fib.search(approximation, 0, 10, self.epsilon)
        return beta


epsilon = 0.00001
x_start = np.array([-10, -48])


def f(X):
    x1, x2 = X
    return 3 * x1 * x1 - 3 * x1 * x2 + 4 * x2 * x2 - 2 * x1 + x2


def dydx1(X):
    x1, x2 = X
    return 6 * x1 - 3 * x2 - 2


def dydx2(X):
    x1, x2 = X
    return -3 * x1 + 8 * x2 + 1


gd = GradientDescentWithFibonacci(f, dydx1, dydx2, epsilon, x_start)
min_f, coordinates = gd.solve()
print("\n\n## My solution with Fibonacci search ##")
print("F =", min_f)
print("X =", coordinates)

print("\n\n## Scipy solution ##")
print(minimize(f, x_start))

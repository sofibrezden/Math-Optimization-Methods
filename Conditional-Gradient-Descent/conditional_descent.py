from scipy.optimize import Bounds, LinearConstraint, minimize
import numpy as np


class Simplex:
    def __init__(self, c, A, b):
        self.c = np.array(c)
        self.A = np.array(A)
        self.b = np.array(b)
        self.m = len(self.b)
        self.n = len(self.c)
        self.basis = []

    def _add_variables(self):
        self.A = np.hstack((self.A, np.eye(self.m)))
        self.c = np.hstack((self.c, np.zeros(self.m)))
        self.basis.extend(list(range(self.n, self.n + self.m)))
        self.n += self.m

    def _construct_tableau(self):
        self._add_variables()
        self.tableau = np.hstack((self.A, self.b.reshape(-1, 1)))
        self.tableau = np.vstack((self.tableau, -np.concatenate((self.c, np.zeros(1)))))

    def _pivot_operation(self, pivot_col, pivot_row):
        self.tableau[pivot_row, :] /= self.tableau[pivot_row, pivot_col]
        for r, _ in enumerate(self.tableau):
            if r != pivot_row:
                self.tableau[r, :] -= self.tableau[r, pivot_col] * self.tableau[pivot_row, :]
        self.basis[pivot_row] = pivot_col

    def _primal_step(self):
        # choose pivot column
        pivot_col = np.argmin(self.tableau[-1, :-1])

        # choose pivot row
        mask = self.tableau[:-1, pivot_col] > 0
        ratios = np.full(mask.shape, np.inf)
        ratios[mask] = self.tableau[:-1, -1][mask] / self.tableau[:-1, pivot_col][mask]
        if np.all(ratios == np.inf):
            raise ValueError("Problem is unbounded")
        pivot_row = np.argmin(ratios)

        # pivot operation
        self._pivot_operation(pivot_col, pivot_row)

    def _dual_step(self):
        # choose pivot row
        pivot_row = np.argmin(self.tableau[:-1, -1])

        # choose pivot column
        mask = self.tableau[pivot_row, :-1] < 0
        ratios = np.full(mask.shape, np.inf)
        ratios[mask] = abs(self.tableau[-1, :-1][mask] / self.tableau[pivot_row, :-1][mask])
        if np.all(ratios == np.inf):
            raise ValueError("Problem is unbounded")
        pivot_col = np.argmin(ratios)

        # pivot operation
        self._pivot_operation(pivot_col, pivot_row)

    def _simplex_algorithm(self):
        self._construct_tableau()
        while np.any(self.tableau[:-1, -1] < 0):
            self._dual_step()
        while np.any(self.tableau[-1, :-1] < 0):
            self._primal_step()

    def solve(self):
        self._simplex_algorithm()

        solution = np.zeros(self.n)
        solution[self.basis] = self.tableau[:-1, -1]
        return solution[:self.n - self.m]


def fibonachi(n):
    i, j = 1, 1
    for _ in range(n):
        i, j = j, i + j
    return i


def get_iteration(vars):
    i, j, c = 1, 1, 0
    while i <= vars:
        i, j = j, i + j
        c += 1
    return c


def fib_method(f, a, b, eps=1e-13):
    # знаходить найменше потрібне число фібоначі
    vars = round((b - a) / eps)

    # знаходить номер того фібоначі числа
    n = get_iteration(vars) + 1

    x1 = a + (fibonachi(n - 2) / fibonachi(n)) * (b - a)
    x2 = a + (fibonachi(n - 1) / fibonachi(n)) * (b - a)
    f_x1 = f(x1)
    f_x2 = f(x2)
    k = 0
    iterations = 0

    while abs(b - a) >= eps:
        iterations += 1
        if f_x1 <= f_x2:
            b, x2 = x2, x1
            x1 = a + (fibonachi(n - k - 3) / fibonachi(n - k - 1)) * (b - a)
            f_x2 = f_x1
            f_x1 = f(x1)
        else:
            a, x1 = x1, x2
            x2 = a + (fibonachi(n - k - 2) / fibonachi(n - k - 1)) * (b - a)
            f_x1 = f_x2
            f_x2 = f(x2)
        k += 1
        # print(f"Iteration {iterations}: a = {a}, b = {b}, x1 = {x1}, x2 = {x2}, f(x1) = {f_x1}, f(x2) = {f_x2}")

    return (a + b) / 2


def conditional_descent(f, df, x0, A, ub, tol):
    x = np.array(x0, dtype=float)
    iterat = 0

    while True:
        grad = np.array(df(x), dtype=float)
        h = Simplex(c=-grad, A=A, b=ub).solve() - x

        def objective_function(b, x=x, h=h):
            return f(x + b * h)

        beta = fib_method(objective_function, 0, 1)
        x_new = x + beta * h
        if f(x) - f(x_new) < tol:
            break
        x = x_new

        iterat += 1
        print("Iteration", iterat)
        print("Value of x", x)
        print("Value of f", f(x))
        print("*" * 80)

    return x


def f(x):
    return x[0] * x[0] + 2 * x[1] * x[1] - 2 * x[0] - 4 * x[1]


def df(x):
    return (
        2 * x[0] - 2,
        4 * x[1] - 4)


A = np.array([
    [1, 4],
    [2, -1]])
b = np.array([2, 12])
x0 = np.array([0, 0])

tolerance = 0.0001
print(conditional_descent(f, df, x0, A, b, tolerance))
print("*" * 80)

# constraint - x >= 0
bounds = Bounds(np.zeros_like(x0), np.inf * np.ones_like(x0))

# constraint - Linear Constraint A @ x <= b
constraints = LinearConstraint(A, ub=b)  # default lb=-inf

solution = minimize(f, x0, bounds=bounds, constraints=(constraints,))
print("Scipy solution:", solution)

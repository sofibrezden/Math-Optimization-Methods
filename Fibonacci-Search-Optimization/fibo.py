import math
from scipy import optimize


def fibonacci(n):
    i = 1
    j = i
    c = 1
    while c < n:
        k = i
        i = j
        j = k + 1
        c += 1
    return i


def get_iteration(n):
    i = 1
    j = i
    c = 0
    while i <= n:
        k = i
        i = j
        j = k + i
        c += 1
    return c


def F(x):
    return 2 * x / math.log(2) - 2 * x ** 2


def fibonacci_search(a, b, eps):
    vars = round((b - a) / eps)
    n = get_iteration(vars) + 1
    x1 = a + (fibonacci(n - 2) / fibonacci(n)) * (b - a)
    x2 = a + (fibonacci(n - 1) / fibonacci(n)) * (b - a)
    y1 = F(x1)
    y2 = F(x2)

    k = 0
    while abs(b - a) >= eps:
        if y1 <= y2:
            a = x1
            x2 = x1
            x1 = a + (fibonacci(n - k + 1) / fibonacci(n - k + 3)) * (b - a)
            y2 = y1
            y1 = F(x1)
        else:
            b = x2
            x2 = x1
            x1 = a + (fibonacci(n - k + 2) / fibonacci(n - k + 3)) * (b - a)
            y1 = y2
            y2 = F(x2)
        k = k + 1
        print(f"a: {a}, b: {b}, f(xmin):{F((b - a) / 2)}")

        xmin = (a + b) / 2
    return xmin


print("My result:", fibonacci_search(3.5, 5, 0.000002))

print("Built-in result:", optimize.fminbound(F, 3.5, 5, xtol=0.000002))

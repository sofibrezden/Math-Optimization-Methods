import numpy as np
from scipy.optimize import linprog
c = [-2,-1]

# Coefficients of the inequality constraints
A = [
    [-3,-1],
    [-4,-3],
    [1,2]
]

# Right-hand side of the inequality constraints
b = [-3,-6,3]

# c = [2,2]
# A = [[-3,2], [-1,-1], [1,0], [0, 1]]
# b = [6,-3,3,5]

# c = [16, 6]
# A = [[2, 3], [4, 1], [6, 7], [0, -1]]
# b = [180, 240, 426, -20]
# c = [2,2]
# A = [[-3, 2], [-1, -1], [1,0], [0, 1]]
# b = [6, -3, 3,5]
max = True


def calcul_rows_cols(table, basis, pivot_col, pivot_row):
    table[pivot_row, :] /= table[pivot_row, pivot_col]
    for r, _ in enumerate(table):
        if r != pivot_row:
            table[r, :] -= table[r, pivot_col] * table[pivot_row, :]  # правило чтрк
    basis[pivot_row] = pivot_col  # вносимо новий базис


# кожне обмеження з вектора b перевіряється на його знак
def canonic_form(A, c, b, n, basis, m):
    A = np.hstack((A, np.eye(m)))
    c = np.hstack((c, np.zeros(m)))#коефіц яких нема ставим 0
    basis.extend(list(range(n, n + m)))
    n += m
    return A, c, b, n, basis, m


def build_table(A, c, b, n, basis, m):
    A, c, b, n, basis, m = canonic_form(A, c, b, n, basis, m)
    table = np.hstack((A, b.reshape(-1, 1)))#додаєм план справа
    table = np.vstack((table, -np.concatenate((c, np.zeros(1)))))
    return table, n, c


def simplex_step(table, basis):
    # choose pivot column
    pivot_col = np.argmin(table[-1, :-1])  # індекс мінімального елемента в заданому масиві

    # choose pivot row
    help = table[:-1, pivot_col] > 0
    # help, що містить значення True для всіх рядків, де відповідний
    # елемент у стовпці pivot_col більше за 0, і значення
    # False в інших випадках.

    q = np.full(help.shape, np.inf)  # ініціалузую інфами оціночний стовпець
    q[help] = table[:-1, -1][help] / table[:-1, pivot_col][help]
    if np.all(q == np.inf):
        print("unbounded")
        exit(0)
    pivot_row = np.argmin(q)

    # pivot operation
    calcul_rows_cols(table, basis, pivot_col, pivot_row)


def dual(table, basis):
    # choose pivot row
    pivot_row = np.argmin(table[:-1, -1])  # min from b

    # choose pivot column
    help = table[pivot_row, :-1] < 0
    q = np.full(help.shape, np.inf)
    a = table[-1, :-1][help]
    b = table[pivot_row, :-1][help]

    q[help] = abs(table[-1, :-1][help] / table[pivot_row, :-1][help])
    if np.all(q == np.inf):
        raise ValueError("unbounded")
    pivot_col = np.argmin(q)

    # pivot operation
    calcul_rows_cols(table, basis, pivot_col, pivot_row)


def solve(c, A, b):
    A = np.array(A)
    c = np.array(c)
    b = np.array(b)
    m = len(b)
    n = len(c)
    basis = []
    table, n, c = build_table(A, c, b, n, basis, m)
    while np.any(table[:-1, -1] < 0):
        dual(table, basis)
    while np.any(table[-1, :-1] < 0):
        simplex_step(table, basis)

    solution = np.zeros(n)
    solution[basis] = table[:-1, -1]
    return solution, solution @ c



if max:
    solution, max_value = solve(c, A, b)
    print()
    print('Dual simplex method')

    print("Solution", solution[:len(c)])
    print('Fmax(x) =', max_value)
    print('----------------------------------')
    print()
    print('----------------------------------')
    print("linprog implementation")
    print()
    c_for_linprog = -np.array(c)
    print(linprog(c_for_linprog, A, b, bounds=(0, None), method='simplex'))
else:

    c_min = -np.array(c)
    solution, min_value = solve(c_min, A, b)
    print()
    print('Dual simplex method')

    print("Solution", solution[:len(c)])
    print('Fmin(x) =', -min_value)
    print('----------------------------------')
    print()
    print('----------------------------------')
    print("linprog implementation")
    print()
    print(linprog(c, A, b, bounds=(0, None), method='simplex'))



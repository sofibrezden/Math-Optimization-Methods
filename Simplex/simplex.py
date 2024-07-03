import numpy as np
from scipy.optimize import linprog

c = [-8, 6, 3]
# #
# #
A = [[-2, 1, 3], [-1, 3, 1]]
b = [-2, 1]

# b = [27, -21, -6]
# A = [[1, 1], [-3, -7], [-1, -2]]

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
M = -100000


def calcul_rows_cols(table, basis, pivot_col, pivot_row):
    table[pivot_row, :] /= table[pivot_row, pivot_col]
    for r, _ in enumerate(table):
        if r != pivot_row:
            table[r, :] -= table[r, pivot_col] * table[pivot_row, :]  # правило чтрк
    basis[pivot_row] = pivot_col  # вносимо новий базис


def slack_variables(A, c, b, n, position):
    v = np.zeros_like(b)
    v[position] = 1
    A = np.column_stack((A, v))
    c = np.append(c, 0)
    n += 1
    return A, c, n


def artif_var(A, c, b, n, position):
    v = np.zeros_like(b)
    v[position] = 1
    A = np.column_stack((A, v))
    c = np.append(c, M)
    n += 1
    return A, c, n


# кожне обмеження з вектора b перевіряється на його знак
def canonic_form(A, c, b, n, basis, artificial_vars):
    for (i, el) in enumerate(b):
        if el >= 0:
            # add slack var
            basis.append(n)
            A, c, n = slack_variables(A, c, b, n, i)
        else:
            # add slack var
            A, c, n = slack_variables(A, c, b, n, i)
            A[i, :] *= -1
            b[i] *= -1

            # add artificial var
            basis.append(n)
            artificial_vars.append(n)
            A, c, n = artif_var(A, c, b, n, i)
    return A, c, n


def build_table(A, c, b, n, basis, artificial_vars):
    A, c, n = canonic_form(A, c, b, n, basis, artificial_vars)
    table = np.hstack((A, b.reshape(-1, 1)))  # ставлю у відповідність елементи b
    init_estimate = table.T @ c[basis] - np.concatenate((c, np.zeros(1)))
    # c[basis] вибирає лише ті коефіцієнти цільової функції,
    # які відповідають базисним змінним.

    table = np.vstack((table, init_estimate))  # вертикальне зєднання
    return table, n, c


def simplex_step(table, basis, artificial_vars):
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


def solve(c, A, b):
    A = np.array(A)
    c = np.array(c)
    b = np.array(b)
    m = len(b)
    n = len(c)
    basis = []
    artificial_vars = []
    table, n, c = build_table(A, c, b, n, basis, artificial_vars)
    while np.any(table[-1, :-1] < 0):
        simplex_step(table, basis, artificial_vars)

    if set(artificial_vars).intersection(set(basis)):
        print("No solution found")
        exit(0)

    solution = np.zeros(n)
    solution[basis] = table[:-1, -1]
    return solution, solution @ c
    # скалярний добуток масивів solution та коефіцієнтів цільової функції c


if max:
    solution, max_value = solve(c, A, b)
    print()
    print('Simplex method')

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
    c_dual = -np.array(b)  # Змінені коефіцієнти цільової функції
    A_dual = -np.array(A).T  # Змінені обмеження (рядки стали стовпцями і навпаки)
    b_dual = np.array(c)  # Змінені обмеження (стовпці стали рядками і навпаки)

    solution_dual, max_value_dual = solve(c_dual, A_dual, b_dual)

    print("Solution for dual problem:", solution_dual[:len(c_dual)])
    print('Fmax(x) for dual problem =', max_value_dual)

    c_min = -np.array(c)
    solution, min_value = solve(c_min, A, b)
    print()
    print('Simplex method')

    print("Solution", solution[:len(c)])
    print('Fmin(x) =', min_value)
    print('----------------------------------')
    print()
    print('----------------------------------')
    print("linprog implementation")
    print()
    print(linprog(c, A, b, bounds=(0, None), method='simplex'))

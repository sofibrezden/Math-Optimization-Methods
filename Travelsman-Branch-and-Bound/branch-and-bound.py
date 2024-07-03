import numpy as np
import pandas as pd
from queue import PriorityQueue
from python_tsp.exact import solve_tsp_dynamic_programming


def make_list(path):
    # Отримуємо кількість вершин у шляху
    len_path = len(path)

    # Видаляємо останню вершину з шляху і додаємо її до списку вершин
    end_node = path.pop()
    path_as_list = [end_node[1]]  # Додаємо кінцеву вершину шляху

    # Починаємо з кінцевої вершини і шукаємо попередні вершини у шляху
    while len(path_as_list) < len_path:
        for node in path:
            # Шукаємо вершину, яка є попередньою за поточною вершиною у шляху
            if node[1] == end_node[0]:
                # Якщо знайдено, додаємо її до списку і оновлюємо поточну вершину
                path_as_list.append(node[0])
                end_node = node
                break  # Виходимо з циклу, коли знайдено вершину

    # Повертаємо список вершин у порядку зворотнього шляху
    return path_as_list[::-1]


class Step:
    def __init__(self, matrix, lower_bound, path):
        self.matrix = matrix
        self.path = path
        self.lower_bound = lower_bound

    # визначаємо на основі чого будем пріоритезувати в черзі
    def __lt__(self, other):
        return self.lower_bound < other.lower_bound

    # редукуємо матрицю
    def reduction(self):
        di = self.matrix.min(axis=1)
        self.matrix = self.matrix.sub(di, axis=0)
        dj = self.matrix.min(axis=0)
        self.matrix = self.matrix.sub(dj, axis=1)
        self.lower_bound += di.sum() + dj.sum()

    # штраф за невключення дуги
    def pivot(self):
        max_value = -np.inf
        pivot = (None, None)
        for i in self.matrix.index:
            for j in self.matrix.columns:
                if self.matrix.loc[i, j] == 0:
                    col = self.matrix.loc[:, j].drop(i)
                    row = self.matrix.drop(j, axis=1).loc[i]
                    sum_of_minimal = col.min() + row.min()
                    # знаходимо найбільший штраф
                    if sum_of_minimal > max_value:
                        max_value = sum_of_minimal
                        pivot = (i, j)

        return max_value, pivot


def get_cost(distances, path):
    return sum([distances.loc[*cell] for cell in path])


def branch_and_bound(distances):
    pq = PriorityQueue()

    optimal_path = [
        (i % len(distances), (i + 1) % len(distances)) for i in range(0, len(distances))
    ]
    optimal_cost = get_cost(distances, optimal_path)

    first_step = Step(distances, 0, [])
    pq.put(first_step)
    # поки черга пріоритетів pq не стане порожньою
    while not pq.empty():
        current = pq.get()
        current.reduction()
        if current.lower_bound >= optimal_cost:
            continue

        value, (i, j) = current.pivot()

        # виключення ребра
        excluded = Step(current.matrix.copy(), current.lower_bound , current.path)
        # ставимо inf для i,j елемента
        if i in excluded.matrix.index and j in excluded.matrix.columns:
            excluded.matrix.loc[i, j] = np.inf
        # додає новий стан до черги пріоритетів pq
        pq.put(excluded)

        # включення ребра видаляєм i row anf j col
        included = current.matrix.drop(i, axis=0).drop(j, axis=1)
        # ставимо inf в зворотню стороно j i  елемент
        if j in included.index and i in included.columns:
            included.loc[j, i] = np.inf
        # формує новий шлях new_path, додавши до поточного шляху
        # current.path нову дугу (i, j)
        new_path = current.path + [(i, j)]

        if len(included) == 2:
            # Знаходиться індекс дуги, яка має значення нескінченності в матриці included
            # inf_index буде містити індекс дуги у вигляді кортежу (рядок_індексу, стовпчик_індексу),
            # де значення в матриці included рівне нескінченності.
            inf_index = included.stack()[np.isinf(included.stack())].index[0]
            # Для кожного рядка перевіряється, чи він не дорівнює індексу рядка
            # inf_index[0], який відповідає рядковому індексу дуги, що має значення нескінченності в матриці.
            for row in included.index:
                if row != inf_index[0]:
                    new_path += [(row, inf_index[1])]
            for col in included.columns:
                if col != inf_index[1]:
                    new_path += [(inf_index[0], col)]

            assert len(new_path) == len(distances)

            new_path_cost = get_cost(distances, new_path)
            if new_path_cost < optimal_cost:
                optimal_cost = new_path_cost
                optimal_path = new_path
        else:
            # додає новий стан до черги пріоритетів pq
            pq.put(Step(included, current.lower_bound, new_path))

    return optimal_path, optimal_cost


matrix = np.array([
    [np.inf, 70, 70, 18, 10],
    [200, np.inf, 20, 18, 13],
    [10, 7, np.inf, 7, 10],
    [19, 12, 19, np.inf, 12],
    [8, 14, 200, 18, np.inf]
])

# matrix = np.array([
#     [np.inf, 11, 21, 6, 8],
#     [13, np.inf, 17, 8, 11],
#     [19, 18, np.inf, 7, 21],
#     [22, 15, 11, np.inf, 17],
#     [32, 4, 12, 6, np.inf]
# ])

# matrix = np.array([
#     [np.inf, 20, 18, 12, 8],
#     [5, np.inf, 14, 7, 11],
#     [12, 18, np.inf, 6, 11],
#     [11, 17, 11, np.inf, 12],
#     [17, 17, 17, 17, np.inf]
# ])
# 4
# matrix = np.array([
#     [np.inf, 10, 5, 14, 19],
#     [8, np.inf, 16, 16, 8],
#     [20, 6, np.inf, 18, 7],
#     [9, 14, 6, np.inf, 10],
#     [2, 13, 13, 12, np.inf]
# ])
# 5
# matrix = np.array([
#     [np.inf, 20, 4, 20, 15],
#     [3, np.inf, 11, 14, 3],
#     [4, 8, np.inf, 14, 2],
#     [13, 12, 4, np.inf, 15],
#     [12, 16, 5, 20, np.inf]
# ])
# 6
# matrix = np.array([
#     [np.inf, 4, 5, 18, 13],
#     [8, np.inf, 3, 15, 2],
#     [18, 19, np.inf, 19, 13],
#     [13, 2, 10, np.inf, 18],
#     [15, 2, 11, 6, np.inf]
# ])
# 7
# matrix = np.array([
#     [np.inf, 5, 10, 19, 14],
#     [10, np.inf, 3, 11, 20],
#     [12, 15, np.inf, 13, 3],
#     [19, 20, 6, np.inf, 12],
#     [19, 9, 5,12, np.inf]
# ])
a = pd.DataFrame(matrix)
res = branch_and_bound(a)
print(f"My Optimal Path: {(res[0])}")
print(f"Minimum Cost: {res[1]}")
print()
print()
res = solve_tsp_dynamic_programming(matrix)
print(f"Optimal Path from python_tsp: {res[0]}")
print(f"Minimum Cost from python_tsp: {res[1]}")

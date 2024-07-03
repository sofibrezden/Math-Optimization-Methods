import pulp
import numpy as np

c = np.array([
    [0.3, 0.4, 0.1, 0.2, 0.4],
    [0.2, 0.3, 0.4, 0.3, 0.2],
    [0.4, 0.3, 0.2, 0.4, 0.3],
])
s = np.array([700, 700, 700])
d = np.array([250, 480, 360, 540, 470])
# s = np.array([10, 20, 35, 45])
# d = np.array([25, 30, 40, 15])
# c = np.array([
#     [1, 3, 3, 7],
#     [8, 6, 2, 6],
#     [4, 7, 7, 3],
#     [5, 2, 4, 5]])
# c = np.array([
#     [2,4,5,1],
#     [2,3,9,4],
#     [3,4,2,5]
# ])
#
# s = np.array([60,70,20])
# d = np.array([40,30,30,50])
# c = np.array([
#     [2, 6, 3, 4, 8],
#     [1, 5, 6, 9, 7],
#     [3, 4, 1, 6, 10]
# ])
#
# s = np.array([400, 300,350])
# d = np.array([20, 34, 16, 10, 25])

# c = np.array([
#         [1, 3, 3, 4],
#         [5, 2, 7, 5],
#         [6, 4, 8, 2],
#         [7, 1, 5, 7]
#     ])
# s = np.array([50, 20, 30, 20])
# d = np.array([40, 30, 25, 15])
prob = pulp.LpProblem("Transportation_Problem", pulp.LpMinimize)

amounts = pulp.LpVariable.dicts("Amount", [(i, j) for i in range(len(s)) for j in range(len(d))], 0, None,
                                pulp.LpInteger)

prob += pulp.lpSum(amounts[(i, j)] * c[i][j] for i in range(len(s)) for j in range(len(d)))

for i in range(len(s)):
    prob += pulp.lpSum(amounts[(i, j)] for j in range(len(d))) <= s[i], f"s_{i}"

for j in range(len(d)):
    prob += pulp.lpSum(amounts[(i, j)] for i in range(len(s))) >= d[j], f"d_{j}"

prob.solve()

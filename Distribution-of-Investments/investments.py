invest = 2
costs = [
    [3, 3, 2, 1, 1],
    [1, 1, 3, 2, 2],
    [1, 3, 1, 4, 2]
]
profits = [
    [0.8, 0.6, 0.7, 0.2, 1.0],
    [0.3, 0.6, 1.1, 1.0, 1.3],
    [0.3, 0.6, 0.4, 1.2, 0.8]
]


def find_max(f):
    max_val = f[0]
    n = 0
    for i in range(len(f)):
        if max_val < f[i]:
            max_val = f[i]
            n = i

    return [max_val, n]


companies = len(costs)
projects = len(costs[0])
amount_left = invest
calculated_profits = [0] * (invest + 1)
temp = []
res = [[] for _ in range(companies)]

for i in range(companies - 1, -1, -1):
    curr = [[0 for _ in range(invest + 1)] for _ in range(invest + 1)]
    for k in range(invest + 1):
        for j in range(k, invest + 1):
            max_profit = 0
            max_cost = 0
            for l in range(len(costs[i])):
                if costs[i][l] == k and profits[i][l] > max_profit:
                    max_cost = costs[i][l]
                    max_profit = profits[i][l]
            curr[j][k] = max_profit + calculated_profits[j - max_cost]
    temp = [find_max(idx) for idx in curr]
    res[i] = [j[1] for j in temp]
    calculated_profits = [j[0] for j in temp]

for i in range(len(res)):
    print("Investment for", i + 1, "company:", res[i][amount_left])
    a = []
    for j in range(len(costs[i])):
        if costs[i][j] == res[i][amount_left]:
            a.append(profits[i][j])
    if a:
        print("Profit for", i + 1, "company:", max(a))
    else:
        print("No profit for", i + 1, "company.")
    amount_left -= res[i][amount_left]

print("Total profit:", round(temp[invest][0], 1))

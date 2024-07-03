import numpy as np
import pulp


class MODISolver:
    def __init__(self, costs, supply, demand):
        assert costs.shape == (*supply.shape, *demand.shape), "Wrong shapes"
        assert supply.sum() == demand.sum(), "The sum of supply should be equal to the sum of demand"

        self.costs = costs.copy()
        self.supply = supply.copy()
        self.demand = demand.copy()

    def min_method(self):
        costs = self.costs.copy()
        solution = np.full(costs.shape, np.nan)

        while True:
            min_i, min_j = np.unravel_index(np.argmin(costs), costs.shape)
            min_value = min(self.supply[min_i], self.demand[min_j])

            if min_value != 0:
                solution[min_i][min_j] = min_value

            self.supply[min_i] -= min_value
            self.demand[min_j] -= min_value
            costs[min_i][min_j] = np.iinfo(np.int32).max

            if self.supply.sum() == 0 or self.demand.sum() == 0:
                break

        return solution

    def calculate_uv(self, solution):
        num_sources, num_destinations = self.costs.shape
        u = np.full(num_sources, np.nan)
        v = np.full(num_destinations, np.nan)
        u[0] = 0
        n, m = solution.shape
        if np.sum(~np.isnan(solution)) != n + m - 1:
            index = np.where(np.isnan(solution))
            solution[index[0][0], index[1][0]] = 0

        while np.isnan(u).any() or np.isnan(v).any():
            for i in range(num_sources):
                for j in range(num_destinations):
                    if not np.isnan(solution[i, j]):
                        if not np.isnan(u[i]) and np.isnan(v[j]):
                            v[j] = self.costs[i, j] - u[i]
                        elif np.isnan(u[i]) and not np.isnan(v[j]):
                            u[i] = self.costs[i, j] - v[j]
        return u, v

    def find_opportunity_costs(self, u, v):
        op_costs = np.zeros_like(self.costs)
        for i in range(len(u)):
            for j in range(len(v)):
                op_costs[i, j] = self.costs[i, j] - (u[i] + v[j])
        return op_costs

    @staticmethod
    def adjust_solution(solution, path):
        min_qty = min(solution[path[i]] for i in range(1, len(path), 2))
        sign = 1
        flag = True
        for cell in path:
            if solution[cell] == min_qty and flag and sign == -1:
                solution[cell] = np.nan
                flag = False
            elif np.isnan(solution[cell]):
                solution[cell] = sign * min_qty
            else:
                solution[cell] += sign * min_qty

            sign *= -1

    @staticmethod
    def find_path(start, solution):
        path, been_there = [start], []
        i, j = start
        vertical, forward = True, True
        while path:
            if vertical:
                i = i + 1 if forward else i - 1
            else:
                j = j + 1 if forward else j - 1

            if (i, j) == start:
                return path

            if 0 <= i < solution.shape[0] and 0 <= j < solution.shape[1]:
                if np.isnan(solution[i, j]) or (i, j) in been_there: continue
                path.append((i, j))
                vertical, forward = not vertical, True
            elif i == solution.shape[0] or j == solution.shape[1]:
                forward = False
                (i, j) = path[-1]
            elif i == -1 or j == -1:
                been_there.append(path.pop())
                if not path:
                    break
                (i, j) = path[-1]
                vertical, forward = not vertical, True

    def solve(self):
        solution = self.min_method()

        while True:
            u, v = self.calculate_uv(solution)
            op_costs = self.find_opportunity_costs(u, v)
            if (op_costs >= 0).all():
                break

            i, j = np.unravel_index(np.argmin(op_costs), op_costs.shape)
            path = self.find_path((i, j), solution)
            if not path:
                break
            self.adjust_solution(solution, path)

        solution[np.isnan(solution)] = 0
        return solution, np.sum(solution * self.costs)


def check_with_pulp(_costs, _supply, _demand):
    prob = pulp.LpProblem("Transportation_Problem", pulp.LpMinimize)

    amounts = pulp.LpVariable.dicts("Amount", [(i, j) for i in range(len(_supply)) for j in range(len(_demand))], 0,
                                    None,
                                    pulp.LpInteger)

    prob += pulp.lpSum(amounts[(i, j)] * _costs[i][j] for i in range(len(_supply)) for j in range(len(_demand)))

    for i in range(len(_supply)):
        prob += pulp.lpSum(amounts[(i, j)] for j in range(len(_demand))) <= _supply[i], f"Supply_{i}"

    for j in range(len(_demand)):
        prob += pulp.lpSum(amounts[(i, j)] for i in range(len(_supply))) >= _demand[j], f"Demand_{j}"

    prob.solve()


def make_close(costs, supply, demand):
    supply_sum, demand_sum = supply.sum(), demand.sum()
    if supply_sum > demand_sum:
        demand = np.append(demand, supply_sum - demand_sum)
        costs = np.column_stack((costs, np.zeros_like(supply)))
    if demand_sum > supply_sum:
        supply = np.append(supply, demand_sum - supply_sum)
        costs = np.row_stack((costs, np.zeros_like(demand)))

    return costs, supply, demand


if __name__ == '__main__':
    c = np.array([
        [0.3, 0.4, 0.1, 0.2, 0.4],
        [0.2, 0.3, 0.4, 0.3, 0.2],
        [0.4, 0.3, 0.2, 0.4, 0.3],
    ])
    s = np.array([700, 700, 700])
    d = np.array([250, 480, 360, 540, 470])

    c, s, d = make_close(c, s, d)

    solver = MODISolver(c, s, d)
    optimized_solution, total_cost = solver.solve()

    print("My Solution:\n", optimized_solution)
    print("Total Transportation Cost:", total_cost)

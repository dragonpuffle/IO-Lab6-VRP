import math

import matplotlib.pyplot as plt
import numpy as np

from src.read_data import VRPData, read_parse_data


class AntColonyAlgorithm:
    def __init__(self, data: VRPData, alpha: float = 0.5, beta: float = 5.0, phi=0.6, q: int = 100,
                 num_ants: int = 10, iterations: int = 100):
        self.optimal_value = data.optimal_value
        self.num_customers = data.num_customers
        self.num_trucks = data.num_trucks
        self.truck_capacity = data.truck_capacity
        self.coords = data.coords
        self.demands = data.demands
        self.start_id = data.start_id

        self.alpha = alpha
        self.beta = beta
        self.phi = phi
        self.q = q
        self.num_ants = num_ants
        self.iterations = iterations

        self.distance_mtx = self.compute_distance_mtx()
        self.pheromones = np.ones((self.num_customers, self.num_customers))

        self.best_solution = None
        self.best_cost = float('inf')


    def compute_distance_mtx(self) -> np.ndarray:
        distance_mtx = np.zeros((self.num_customers, self.num_customers))
        for i in range(self.num_customers):
            for j in range(self.num_customers):
                xi, yi = self.coords[i]
                xj, yj = self.coords[j]
                distance_mtx[i][j] = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
        return distance_mtx


    def select_next_node(self, node: int, candidates: list):
        pheromone = self.pheromones[node, candidates]
        attractiveness = 1 / (self.distance_mtx[node, candidates] + 1e-7)

        weights = (pheromone ** self.alpha) * (attractiveness ** self.beta)
        probs = weights / np.sum(weights)

        return np.random.choice(candidates, p=probs)


    def ants_routes(self):
        unvisited = set(range(self.num_customers))
        unvisited.remove(self.start_id)

        routes = []
        total_cost = 0

        for _ in range(self.num_trucks):
            current_route = [self.start_id]
            current_route_cost = 0
            current_node = self.start_id
            current_load = 0

            while True:
                candidates = [j for j in unvisited if self.demands[j] + current_load <= self.truck_capacity]
                if not candidates:
                    break

                next_node = self.select_next_node(current_node, candidates)
                unvisited.remove(next_node)

                current_route_cost += self.distance_mtx[current_node][next_node]
                current_load += self.demands[next_node]
                current_route.append(int(next_node))
                current_node = next_node

            current_route_cost += self.distance_mtx[current_node][self.start_id]
            current_route.append(self.start_id)

            routes.append(current_route)
            total_cost += current_route_cost

            if not unvisited:
                break

        return routes, total_cost


    def update_pheromones(self, all_solutions: list):
        self.pheromones *= self.phi
        best_solution, best_cost = min(all_solutions, key=lambda x: x[1])
        for route in best_solution:
            for i in range(len(route) - 1):
                a, b = route[i], route[i + 1]
                self.pheromones[a, b] += self.q / best_cost


    def inc_routes(self):
        self.best_solution = [[node_id + 1 for node_id in route] for route in self.best_solution]


    def solve(self, visualize:bool = False):
        for iteration in range(self.iterations):
            all_solutions = []

            # 1 создание муравья
            for _ in range(self.num_ants):
                # 2 поиск решения
                solution, cost = self.ants_routes()
                all_solutions.append((solution, cost))

                if cost < self.best_cost:
                    self.best_cost = cost
                    self.best_solution = solution

            # 3 обновление феромона
            self.update_pheromones(all_solutions)
        if visualize:
            self.visualize() # always before inc_routes!!!
        self.inc_routes()
        return self.best_solution, self.best_cost


    def visualize(self):
        if not self.best_solution:
            print('No solution found')
            return

        plt.figure(figsize=(10, 10))

        for node_id, (x, y) in self.coords.items():
            color = 'red' if node_id == self.start_id else 'blue'
            plt.scatter(x, y, c=color, s=50)

        colors = plt.colormaps.get_cmap('tab10')
        for route_id, route in enumerate(self.best_solution):
            for node_id in range(len(route) - 1):
                a, b = route[node_id], route[node_id + 1]
                x1, y1 = self.coords[a]
                x2, y2 = self.coords[b]
                plt.arrow(x1, y1, x2-x1, y2-y1, color=colors(route_id), length_includes_head=True, head_width=1.5, alpha=0.5)

        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    data1 = read_parse_data('benchmarks/A/A-n32-k5.vrp')
    alg = AntColonyAlgorithm(data1, alpha=0.5, beta=5.0, phi=0.3, q=100, num_ants=20, iterations=300)
    solution1, cost1 = alg.solve(True)
    print('solution:', solution1, 'cost:', cost1)
    print('optimal value:', alg.optimal_value)

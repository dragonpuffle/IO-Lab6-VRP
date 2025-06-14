import os
import time
import pandas as pd
from src.ant_alg import *


class Benchmark:
    def __init__(self, algorithm, bench_dirs: list, runs: int = 3):
        self.algorithm = algorithm
        self.runs = runs
        self.bench_dirs = bench_dirs
        self.results = []

    def run_all(self):
        files = [f for f in os.listdir(self.bench_dirs[0])]
        for file in sorted(files):
            print(f'Benchmarking {file} ...')
            full_path = os.path.join(self.bench_dirs[0], file)
            self.run_one(full_path, file)

        files = [f for f in os.listdir(self.bench_dirs[1])]
        for file in sorted(files):
            print(f'Benchmarking {file} ...')
            full_path = os.path.join(self.bench_dirs[1], file)
            self.run_one(full_path, file)

        df = pd.DataFrame(self.results).sort_values(by=['benchmark', 'alg'], ascending=True)
        df.to_csv('results.csv', index=False)

    def run_one(self, path: str, benchmark: str):
        data = read_parse_data(path)

        best_total_cost = float('inf')
        best_solution = []
        total_time = 0
        optimal_value = 0
        for _ in range(self.runs):
            alg = AntColonyAlgorithm(data, alpha=0.5, beta=5.0, phi=0.3, q=100, num_ants=10, iterations=100)
            optimal_value = alg.optimal_value
            start = time.time()

            solution, cost = alg.solve(False)

            total_time += time.time() - start
            if cost < best_total_cost:
                best_total_cost = cost
                best_solution = solution.copy()
        avg_time = total_time / self.runs
        self.results.append({
            'benchmark': benchmark,
            'alg': self.algorithm.__name__,
            'best_total_cost': round(best_total_cost, 7),
            'avg_time_sec': round(avg_time, 7),
            'solution': " ".join(map(str, best_solution)),
            'optimal_value': optimal_value,
            'diff': best_total_cost - optimal_value,
            'percent_diff': round(((best_total_cost - optimal_value) / optimal_value) * 100, 7)
        })
        self.save_solution("results/" + benchmark[:-3] + ".sol", best_solution, best_total_cost)

    def save_solution(self, file_path: str, solution, cost) -> None:
        with open(file_path, 'w') as file:
            for index, route in enumerate(solution):
                file.write(f"Route #{index + 1}: {" ".join(map(str, route))}")
                file.write("\n")
            file.write(f"cost {cost}")
            file.write("\n")


if __name__ == '__main__':
    benchmark = Benchmark(AntColonyAlgorithm, ['benchmarks/A', 'benchmarks/B'], 5)
    benchmark.run_all()

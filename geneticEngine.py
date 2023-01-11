import struct
from typing import Callable
import numpy as np
from numpy import ndarray, count_nonzero, sin
from numpy.random import rand, uniform, normal, choice
from tqdm import tqdm


def f(x: ndarray, *args) -> float:
    return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2


def generate_genotype(n_features: int, feature_size=32, divisor_size=32) -> str:
    divisor = ''.join([np.random.choice(['0', '1']) for _ in range(divisor_size)])
    feats = ''.join([np.random.choice(['0', '1']) for _ in range(n_features * feature_size)])
    ret = divisor + feats
    return ret


def float_to_bin(num):
    return bin(struct.unpack('!I', struct.pack('!f', num))[0])[2:].zfill(32)


def bin_to_float(binary):
    return struct.unpack('!f', struct.pack('!I', int(binary, 2)))[0]


def inv_chr(string: str, position: int) -> str:
    if int(string[position]) == 1:
        string = string[:position] + '0' + string[position + 1:]
    else:
        string = string[:position] + '1' + string[position + 1:]
    return string


class GeneticUnitFloat:
    def __init__(self, cost_function: Callable, genotype: str = None, n_features: int = 2, feature_size: int = 32,
                 n_cut_points: int = 1, n_mutate_points: int = 1, cost_function_args=None,
                 divisor_size: int = 32):
        if genotype is None:
            self.genotype = generate_genotype(n_features, feature_size, divisor_size)
        else:
            self.genotype = genotype

        self.divisor_size = divisor_size
        self.n_features = n_features
        self.cost_function = cost_function
        self.cost_function_args = cost_function_args if cost_function_args is not None else ()
        self.feature_size = feature_size
        self.n_cut_points = n_cut_points
        self.n_mutate_points = n_mutate_points

    @property
    def fitness(self):
        try:
            return self._fitness
        except AttributeError:
            self._fitness = -self.cost_function(self.phenotype, *self.cost_function_args)
        return self._fitness

    @property
    def phenotype(self) -> np.ndarray:
        divisor = int(self.genotype[:self.divisor_size], 2)
        feats = self.genotype[self.divisor_size:]
        return np.array([int(feats[i * self.feature_size:i * self.feature_size + self.feature_size],
                             2) / divisor for i in
                         range(self.n_features)])

    def cross(self, other) -> tuple:
        if len(self.genotype) != len(self.genotype):
            raise ValueError("Different genotype sizes")
        cut_points = np.random.choice([i for i in range(self.n_features * self.feature_size + self.divisor_size)],
                                      size=self.n_cut_points, replace=False)
        cut_points = sorted(cut_points)
        cut_points = [0] + cut_points + [self.n_features * self.feature_size + self.divisor_size]
        a = ''
        b = ''
        for i in range(len(cut_points) - 1):
            if i % 2 == 0:
                a += self.genotype[cut_points[i]:cut_points[i + 1]]
                b += other.genotype[cut_points[i]:cut_points[i + 1]]
            else:
                a += other.genotype[cut_points[i]:cut_points[i + 1]]
                b += self.genotype[cut_points[i]:cut_points[i + 1]]

        return GeneticUnitFloat(self.cost_function, a, self.n_features, self.feature_size, self.n_cut_points,
                                self.n_mutate_points, self.cost_function_args, self.divisor_size), \
               GeneticUnitFloat(self.cost_function, b, self.n_features, self.feature_size, self.n_cut_points,
                                self.n_mutate_points, self.cost_function_args, self.divisor_size)

    def mutate(self):
        pivots = np.random.choice([i for i in range(self.n_features * self.feature_size)],
                                  size=self.n_cut_points,
                                  replace=False)
        for i in pivots:
            self.genotype = inv_chr(self.genotype, i)
        return self

    def __str__(self):
        return f'{str(self.phenotype)}, div={int(self.genotype[:self.divisor_size], 2)}'

    def __repr__(self):
        return f'{str(self.phenotype)}, div={int(self.genotype[:self.divisor_size], 2)}'


class GeneticEngine:
    def __init__(self, cost_function: Callable, n_units: int = 256, crossover_prob: float = 0.85,
                 mutation_prob: float = 0.25, n_features: int = 2, feature_size: int = 32,
                 n_cut_points: int = 1, n_mutate_points: int = 1,
                 n_jobs: int = 1, cost_function_args=None):
        self.n_units = n_units
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.n_features = n_features
        self.feature_size = feature_size
        self.n_cut_points = n_cut_points
        self.n_mutate_points = n_mutate_points
        self.cost_function = cost_function
        self.cost_function_args = cost_function_args
        self.n_jobs = n_jobs
        self.population = [GeneticUnitFloat(self.cost_function, n_features=self.n_features,
                                            feature_size=self.feature_size,
                                            n_cut_points=self.n_cut_points,
                                            n_mutate_points=self.n_mutate_points,
                                            cost_function_args=self.cost_function_args) for _ in range(self.n_units)]

    @staticmethod
    def _softmax(x: ndarray) -> ndarray:
        return np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum()

    def run(self, n_generations: int = 100, verbose: bool = True) -> None:
        title = f"Current best fitness: N/A, with phenotype: N/A"
        pbar = tqdm(range(n_generations), disable=not verbose, desc=title)
        for _ in pbar:
            fitness = np.array([unit.fitness for unit in self.population])
            probs = self._softmax(fitness)

            new_population = []
            for j in range(self.n_units // 2):
                if count_nonzero(probs) > self.n_units // 4:
                    choices = choice(self.population, 2, p=probs, replace=False)
                else:
                    choices = choice(self.population, 2, replace=False)

                parent_a, parent_b = choices[0], choices[1]

                if np.random.random() < self.crossover_prob:
                    a, b = parent_a.cross(parent_b)
                else:
                    a, b = parent_a, parent_b

                if np.random.random() < self.mutation_prob:
                    a.mutate()
                if np.random.random() < self.mutation_prob:
                    b.mutate()
                new_population.append(a)
                new_population.append(b)

            self.population = new_population

            fitness = np.array([unit.fitness for unit in self.population])
            best_fitness = max(fitness)
            # best_phenotype = self.population[np.argmax(fitness)].phenotype
            title = f"Current best fitness: {-best_fitness:.6f}"
            pbar.set_description(title)

    def get_best(self):
        fitness = np.array([unit.fitness for unit in self.population])
        best_fitness = max(fitness)
        best_phenotype = self.population[np.argmax(fitness)].phenotype
        return best_phenotype, -best_fitness


if __name__ == '__main__':
    engine = GeneticEngine(cost_function=f, n_units=256, n_features=2, feature_size=32,
                           n_cut_points=3, n_mutate_points=2, n_jobs=6,
                           mutation_prob=0.1, crossover_prob=0.5)
    engine.run(n_generations=1000)
    print(engine.get_best())

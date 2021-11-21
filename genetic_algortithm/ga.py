import random as rd
from operator import attrgetter


# константы генетического алгоритма
POPULATION_SIZE = 200  # количество индивидуумов в популяции
P_CROSSOVER = 0.9  # вероятность скрещивания
P_MUTATION = 0.15  # вероятность мутации индивидуума
MAX_GENERATIONS = 50  # максимальное количество поколений


class Individual:
    def __init__(self, gene):
        self.fitness = [0]
        self.gene = gene

    def __repr__(self):
        return str(self.fitness)

    def __lt__(self, other):
        return self.fitness < other.fitness


def fitness_min(individual):
    return -(individual.gene ** 2 - 4)


def individual_creator():
    return Individual(rd.randint(-100, 101))


def population_creator(n=0):
    return list([individual_creator() for i in range(n)])


population = population_creator(n=POPULATION_SIZE)

fitnessValues = list(map(fitness_min, population))

for individual, fitnessValue in zip(population, fitnessValues):
    individual.fitness = fitnessValue

maxFitnessValues = []
meanFitnessValues = []


def clone(value):
    ind = Individual(value.gene)
    ind.fitness = value.fitness
    return ind


def sel_tournament(population, p_len):
    offspring = []
    for n in range(p_len):
        # t = 2
        i1 = i2 = 0
        while i1 == i2:
            i1, i2 = rd.randint(0, p_len - 1), rd.randint(0, p_len - 1)

        offspring.append(max([population[i1], population[i2]], key=attrgetter('fitness')))

    return offspring


def mutation(mutant):
    mutant.fitness += rd.randint(-1, 2)


def blx_alpha(cmin, cmax, k=0.2):
    delta_k = int((cmax-cmin) * k)
    start = cmin - delta_k
    stop = cmax + delta_k
    return rd.randint(start, stop)


def crossing(child1, child2):
    s = sorted([child1, child2])
    child1.gene = blx_alpha(s[0].fitness, s[1].fitness)


fitnessValues = [individual.fitness for individual in population]


# цикл поколений
generationCounter = 0
while generationCounter < MAX_GENERATIONS:
    generationCounter += 1
    offspring = sel_tournament(population, len(population))
    offspring = list(map(clone, offspring))

    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if rd.random() < P_CROSSOVER:
            crossing(child1, child2)

    for mutant in offspring:
        if rd.random() < P_MUTATION:
            mutation(mutant)

    freshFitnessValues = list(map(fitness_min, offspring))
    # занесение в поле класса individual значения приспособленности
    for individual, fitnessValue in zip(offspring, freshFitnessValues):
        individual.fitness = fitnessValue

    population[:] = offspring

    fitnessValues = [ind.fitness for ind in population]

    maxFitness = max(fitnessValues)
    meanFitness = sum(fitnessValues) / len(population)
    maxFitnessValues.append(maxFitness)
    meanFitnessValues.append(meanFitness)
    print(f"Поколение {generationCounter}:  Средняя приспособ.= {meanFitness}")


print(population)
print(meanFitnessValues)
# genetic_tuner.py
import random

def mutate(hyperparam):
    """
    hyperparam: [learning_rate, batch_size]
    각 값에 대해 ±10% 범위 내에서 돌연변이 적용
    """
    lr, bs = hyperparam
    factor_lr = random.choice([-1, 0, 1])
    factor_bs = random.choice([-1, 0, 1])
    new_lr = lr + ((lr * factor_lr) / 10)
    new_bs = bs + ((bs * factor_bs) / 10)
    return [new_lr, new_bs]

def crossover(hyperparam_list):
    new_params = []
    n = len(hyperparam_list)
    if n >= 2:
        new_params.append(hyperparam_list[0])
        new_params.append(hyperparam_list[1])
    else:
        new_params.extend(hyperparam_list)
    for k in range(2, n):
        parentA = random.choice(hyperparam_list)
        parentB = random.choice(hyperparam_list)
        # 평균 계산
        child = [(parentA[i] + parentB[i]) / 2 for i in range(len(parentA))]
        child = mutate(child)
        new_params.append(child)
    return new_params

def evolve(losses, hyperparam_list):
    paired = list(zip(losses, hyperparam_list))
    paired.sort(key=lambda x: x[0])
    sorted_params = [p[1] for p in paired]
    return crossover(sorted_params)

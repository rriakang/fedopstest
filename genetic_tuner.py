# genetic_tuner.py
import random

def mutate(eta):
    """
    학습률 eta에 대해 돌연변이를 적용합니다.
    인자로 선택된 factor ( -1, 0, 1 중 랜덤)를 이용해 eta의 10% 범위 내 증감을 반영합니다.
    """
    factor = random.choice([-1, 0, 1])
    new_eta = eta + ((eta * factor) / 10)
    return new_eta

def crossover(eta_list):
    """
    eta_list에 대해 부모 두 개를 선택하여 교차 및 돌연변이 연산을 수행합니다.
    첫 두 개의 개체는 그대로 유지하고, 이후부터는 부모 평균에 mutate를 적용합니다.
    """
    new_eta = []
    n = len(eta_list)
    # 우선 상위 두 개는 그대로 유지
    if n >= 2:
        new_eta.append(eta_list[0])
        new_eta.append(eta_list[1])
    else:
        new_eta.extend(eta_list)
    
    for k in range(2, n):
        parentA = random.choice(eta_list)
        parentB = random.choice(eta_list)
        child = mutate((parentA + parentB) / 2)
        new_eta.append(child)
    return new_eta

def evolve(losses, etas):
    """
    주어진 손실과 해당 학습률 리스트에 대해 오름차순 정렬한 후,
    crossover 연산을 적용해 새로운 학습률 후보 집합을 생성합니다.
    낮은 loss가 더 좋은 hyperparameter임을 전제로 정렬합니다.
    """
    paired = list(zip(losses, etas))
    paired.sort(key=lambda x: x[0])
    sorted_etas = [p[1] for p in paired]
    return crossover(sorted_etas)

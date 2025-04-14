# client_main.py
import random
import hydra
from hydra.utils import instantiate
import numpy as np
import torch
import copy
import data_preparation
import models

from fedops.client import client_utils
from fedops.client.app import FLClientTask
import logging
from omegaconf import DictConfig, OmegaConf

def local_training(model, train_loader, hyperparams, epochs, cfg):
    """
    각 후보 하이퍼파라미터(학습률) candidate별로 1 epoch(또는 epochs) 학습을 진행하여 손실값을 비교.
    각 후보에 대해 optimizer를 새로 생성하여 모델을 복사한 뒤 학습.
    최종적으로 loss가 최소인 후보와 해당 loss, 최종 모델 가중치(state_dict)를 선택.
    """
    candidate_results = []
    for hp in hyperparams:
        temp_model = copy.deepcopy(model)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(temp_model.parameters(), lr=hp)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        temp_model.to(device)
        temp_model.train()
        total_loss = 0.0
        for epoch in range(epochs):
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = temp_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        avg_loss = total_loss / (len(train_loader) * epochs)
        candidate_results.append((hp, avg_loss))
    best_candidate = min(candidate_results, key=lambda x: x[1])
    return best_candidate  # (best_hp, best_loss)

@hydra.main(config_path="./conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # 로깅 설정
    handlers_list = [logging.StreamHandler()]
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s [%(levelname)8.8s] %(message)s",
                        handlers=handlers_list)
    logger = logging.getLogger(__name__)
    
    # 랜덤 시드 설정
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    
    print(OmegaConf.to_yaml(cfg))
    
    # 데이터 로드
    train_loader, val_loader, test_loader = data_preparation.load_partition(
        dataset=cfg.dataset.name, 
        validation_split=cfg.dataset.validation_split, 
        batch_size=cfg.batch_size
    )
    logger.info('Data loaded')
    
    # 모델 및 학습/평가 함수 구성
    model = instantiate(cfg.model)
    model_type = cfg.model_type
    model_name = type(model).__name__
    train_torch = models.train_torch()
    test_torch = models.test_torch()
    
    # 로컬 모델 파일 다운로드 (있을 경우)
    task_id = cfg.task_id
    local_list = client_utils.local_model_directory(task_id)
    if local_list:
        logger.info('Downloading latest local model')
        model = client_utils.download_local_model(model_type=model_type,
                                                  task_id=task_id,
                                                  listdir=local_list,
                                                  model=model)
    
    # 서버로부터 broadcast된 하이퍼파라미터 집합이 있다면 (없으면 config의 후보 사용)
    hyperparams = cfg.get("hyperparams", [0.001, 0.005, 0.01])
    
    # 각 후보 하이퍼파라미터에 대해 로컬 1 epoch 학습을 진행하여 최적 후보 선정
    best_hp, best_loss = local_training(model, train_loader, hyperparams, epochs=1, cfg=cfg)
    logger.info(f"Selected best hyperparameter: {best_hp} with loss: {best_loss}")
    
    # registration 딕셔너리에 학습 결과 및 선택된 하이퍼파라미터 추가 (서버 전송용)
    registration = {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "model": model,
        "model_name": model_name,
        "train_torch": train_torch,
        "test_torch": test_torch,
        "best_hp": best_hp,
        "best_loss": best_loss
    }
    
    # FLClientTask를 초기화하여 서버와 통신 (실제 FedOps의 내부 동작에 따라 client가 업데이트 전송)
    fl_client = FLClientTask(cfg, registration)
    fl_client.start()

if __name__ == "__main__":
    main()

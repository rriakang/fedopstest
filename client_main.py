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
    각 후보 하이퍼파라미터(예: 학습률 또는 [학습률, 배치크기] 형태) candidate별로
    1 epoch(또는 epochs) 학습을 진행하여 손실값을 비교합니다.
    각 후보마다 optimizer를 새로 생성하고 모델 복사본에서 학습한 뒤, 
    최종적으로 평균 손실(loss)이 가장 낮은 후보를 선택합니다.
    """
    from omegaconf import OmegaConf  # Hydra config를 일반 리스트로 변환하기 위해 import
    candidate_results = []
    for hp in hyperparams:
        # Hydra ListConfig 객체를 네이티브 파이썬 객체로 변환
        hp_native = OmegaConf.to_container(hp, resolve=True)
        # hp가 리스트나 튜플인 경우 첫 번째 요소(학습률)를 사용
        if isinstance(hp_native, (list, tuple)):
            lr_val = float(hp_native[0])
        else:
            lr_val = float(hp_native)
        temp_model = copy.deepcopy(model)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(temp_model.parameters(), lr=lr_val)
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
        candidate_results.append((hp_native, avg_loss))
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
    
    # 서버로부터 broadcast된 하이퍼파라미터 후보 집합이 없으면 config의 후보 사용
    hyperparams = cfg.get("hyperparams", [0.001, 0.005, 0.01])
    
    # 각 후보에 대해 로컬 1 epoch 학습을 진행하여 최적 후보(손실 최소)를 선택
    best_hp, best_loss = local_training(model, train_loader, hyperparams, epochs=1, cfg=cfg)
    logger.info(f"Selected best hyperparameter: {best_hp} with loss: {best_loss}")
    
    # registration 딕셔너리에 학습 결과 및 선택된 후보 하이퍼파라미터 추가 (서버 전송용)
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
    
    # FLClientTask를 초기화하여 서버와 통신 (실제 FedOps 내부 동작에 따라 클라이언트 업데이트 전송)
    fl_client = FLClientTask(cfg, registration)
    fl_client.start()

if __name__ == "__main__":
    main()

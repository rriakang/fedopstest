# server_main.py
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
import models
import data_preparation
from server.app import GeneticFLServer  # server/app.py에 정의된 GeneticFLServer 가져오기
import torch

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # 모델 초기화
    model = instantiate(cfg.model)
    model_type = cfg.model_type
    model_name = type(model).__name__
    gl_test_torch = models.test_torch()   # 평가 함수 (PyTorch)
    gl_val_loader = data_preparation.gl_model_torch_validation(batch_size=cfg.batch_size)
    # 구성 파일에 정의된 초기 후보 하이퍼파라미터 사용 (예: [[0.001, 128], [0.005, 64], [0.01, 32]])
    initial_hyperparams = cfg.hyperparams
    fl_server = GeneticFLServer(
        cfg=cfg,
        model=model,
        model_name=model_name,
        model_type=model_type,
        gl_val_loader=gl_val_loader,
        test_torch=gl_test_torch,
        initial_hyperparams=initial_hyperparams,
    )
    fl_server.start()

if __name__ == "__main__":
    main()

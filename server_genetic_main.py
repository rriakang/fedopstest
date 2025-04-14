# server_genetic_main.py
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
import models
import data_preparation
from genetic_tuner import evolve
import torch

# 아래 GeneticFLServer는 FLServer의 기능에 유전 하이퍼파라미터 튜닝 기능을 추가한 예제 클래스입니다.
# 실제 FedOps의 FLServer 클래스를 상속받아 적절히 메서드를 재정의하는 방식으로 구성합니다.
class GeneticFLServer:
    def __init__(self, cfg: DictConfig, model, model_name, model_type, gl_val_loader, test_torch, initial_hyperparams):
        self.cfg = cfg
        self.model = model
        self.model_name = model_name
        self.model_type = model_type
        self.gl_val_loader = gl_val_loader
        self.test_torch = test_torch
        # 초기 후보 하이퍼파라미터 (학습률 집합)
        self.hyperparams = initial_hyperparams  
        # 클라이언트로부터 전달받은 각 업데이트(모델 가중치, 손실, 선택된 하이퍼파라미터) 저장 리스트
        self.client_updates = []

    def get_model_weights(self):
        # 모델의 state_dict를 반환 (필요에 따라 GPU/CPU 이동 처리)
        return self.model.state_dict()

    def set_model_weights(self, new_state_dict):
        self.model.load_state_dict(new_state_dict)

    def send_to_clients(self, broadcast_data):
        """
        서버가 모델 가중치와 하이퍼파라미터 집합을 클라이언트에 전송하는 함수
        (여기서는 자세한 네트워크 통신 코드는 생략 – 실제 환경에 맞게 구현할 것)
        """
        print("Broadcasting model and hyperparameters to clients:")
        print(broadcast_data)

    def collect_client_updates(self):
        """
        클라이언트들로부터 업데이트를 모으는 함수.
        각 업데이트는 아래와 같은 딕셔너리 형태라고 가정합니다.
            { 'weights': state_dict, 'loss': loss_value, 'hyperparam': selected_learning_rate }
        실제 구현에서는 네트워크 통신이나 메시지 큐를 통한 동기화가 필요합니다.
        여기서는 예제 데이터를 생성합니다.
        """
        # 예제: 테스트를 위해 임의의 클라이언트 업데이트 1건을 생성
        dummy_update = {
            "weights": self.get_model_weights(),  # 실제 클라이언트가 보낸 모델 가중치
            "loss": 0.25,                         # 클라이언트 측 학습 손실 (예)
            "hyperparam": self.hyperparams[0]     # 클라이언트가 선택한 최적 학습률 (예)
        }
        print("Collecting client updates ...")
        return [dummy_update]  # 실제 환경에서는 여러 클라이언트의 결과 집합

    def aggregate_client_updates(self, client_updates):
        """
        클라이언트 업데이트(모델 가중치, 손실, 하이퍼파라미터)를 집계하고,
        유전 최적화 알고리즘을 통해 하이퍼파라미터 후보를 진화시킵니다.
        """
        total_weights = None
        losses = []
        hyperparams = []
        for update in client_updates:
            weights = update['weights']
            loss = update['loss']
            hyperparam = update['hyperparam']
            losses.append(loss)
            hyperparams.append(hyperparam)
            if total_weights is None:
                # state_dict의 복사
                total_weights = {k: v.clone() for k, v in weights.items()}
            else:
                for k in total_weights:
                    total_weights[k] += weights[k]
        # 단순 평균: 클라이언트 수로 나누기
        for k in total_weights:
            total_weights[k] = total_weights[k] / len(client_updates)
        # 유전 최적화 적용 – 손실 기준으로 정렬 후 새로운 후보 집합 산출
        new_hyperparams = evolve(losses, hyperparams)
        print(f"Evolved hyperparameters: {new_hyperparams}")
        self.hyperparams = new_hyperparams
        return total_weights

    def evaluate_global_model(self):
        """
        글로벌 모델을 검증 데이터셋(gl_val_loader)으로 평가하는 함수.
        models.test_torch()에서 반환하는 튜플(average_loss, accuracy, metrics)을 그대로 이용합니다.
        """
        test_func = self.test_torch
        loss, acc, metrics = test_func(self.model, self.gl_val_loader, self.cfg)
        return loss, acc, metrics

    def broadcast_model_and_hyperparams(self):
        """
        현재 글로벌 모델과 하이퍼파라미터 집합을 클라이언트에 브로드캐스트합니다.
        """
        broadcast_data = {
            'model_weights': self.get_model_weights(),
            'hyperparams': self.hyperparams
        }
        self.send_to_clients(broadcast_data)

    def train_round(self, round_number):
        # 현재 라운드 시작 시 글로벌 모델과 후보 하이퍼파라미터를 브로드캐스트
        self.broadcast_model_and_hyperparams()
        # 클라이언트로부터 업데이트 수집 (실제 환경에서는 동기화)
        client_updates = self.collect_client_updates()
        # 클라이언트 업데이트를 집계하고 새로운 글로벌 모델과 하이퍼파라미터 집합 도출
        new_global_weights = self.aggregate_client_updates(client_updates)
        self.set_model_weights(new_global_weights)
        loss, acc, _ = self.evaluate_global_model()
        print(f"Round {round_number} - Global Model => Loss: {loss:.4f}, Accuracy: {acc:.4f}")

    def start(self):
        rounds = self.cfg.num_rounds
        for r in range(1, rounds + 1):
            print(f"\n=== Starting Training Round {r} ===")
            self.train_round(r)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # 모델 초기화
    model = instantiate(cfg.model)
    model_type = cfg.model_type
    model_name = type(model).__name__
    gl_test_torch = models.test_torch()  # 평가 함수
    gl_val_loader = data_preparation.gl_model_torch_validation(batch_size=cfg.batch_size)
    # config에 정의된 초기 하이퍼파라미터 후보 사용
    initial_hyperparams = cfg.hyperparams  
    # GeneticFLServer 인스턴스 생성 및 시작
    fl_server = GeneticFLServer(cfg=cfg, model=model, model_name=model_name, model_type=model_type,
                                gl_val_loader=gl_val_loader, test_torch=gl_test_torch,
                                initial_hyperparams=initial_hyperparams)
    fl_server.start()


if __name__ == "__main__":
    main()

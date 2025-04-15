#server/app.py
import logging
from typing import Dict, Optional, Tuple
import flwr as fl
import datetime
import os
import json
import time
import numpy as np
import shutil
from . import server_api
from . import server_utils
from collections import OrderedDict
from hydra.utils import instantiate

import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
import models
import data_preparation
from genetic_tuner import evolve
import torch
import numpy as np
from sklearn.cluster import DBSCAN  # DBSCAN 클러스터링을 위해 추가

# TF warning log filtering
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)8.8s] %(message)s",
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)


class FLServer():
    def __init__(self, cfg, model, model_name, model_type, gl_val_loader=None, x_val=None, y_val=None, test_torch=None):
        
        self.task_id = os.environ.get('TASK_ID') # Set FL Task ID

        self.server = server_utils.FLServerStatus() # Set FLServerStatus class
        self.model_type = model_type
        self.cfg = cfg
        self.strategy = cfg.server.strategy
        
        self.batch_size = int(cfg.batch_size)
        self.local_epochs = int(cfg.num_epochs)
        self.num_rounds = int(cfg.num_rounds)

        self.init_model = model
        self.init_model_name = model_name
        self.next_model = None
        self.next_model_name = None
        
        if self.model_type=="Tensorflow":
            self.x_val = x_val
            self.y_val = y_val  
               

        elif self.model_type == "Pytorch":
            self.gl_val_loader = gl_val_loader
            self.test_torch = test_torch

        elif self.model_type == "Huggingface":
            pass


    def init_gl_model_registration(self, model, gl_model_name) -> None:
        logging.info(f'last_gl_model_v: {self.server.last_gl_model_v}')

        if not model:

            logging.info('init global model making')
            init_model, model_name = self.init_model, self.init_model_name
            print(f'init_gl_model_name: {model_name}')

            self.fl_server_start(init_model, model_name)
            return model_name


        else:
            logging.info('load last global model')
            print(f'last_gl_model_name: {gl_model_name}')

            self.fl_server_start(model, gl_model_name)
            return gl_model_name


    def fl_server_start(self, model, model_name):
        # Load and compile model for
        # 1. server-side parameter initialization
        # 2. server-side parameter evaluation

        model_parameters = None # Init model_parametes variable
        
        if self.model_type == "Tensorflow":
            model_parameters = model.get_weights()
        elif self.model_type == "Pytorch":
            model_parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]
        elif self.model_type == "Huggingface":
            json_path = "./parameter_shapes.json"
            model_parameters = server_utils.load_initial_parameters_from_shape(json_path)

        strategy = instantiate(
            self.strategy,
            initial_parameters=fl.common.ndarrays_to_parameters(model_parameters),
            evaluate_fn=self.get_eval_fn(model, model_name),
            on_fit_config_fn=self.fit_config,
            on_evaluate_config_fn=self.evaluate_config,
        )
        
        # Start Flower server (SSL-enabled) for four rounds of federated learning
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=self.num_rounds),
            strategy=strategy,
        )


    def get_eval_fn(self, model, model_name):
        """Return an evaluation function for server-side evaluation."""
        # Load data and model here to avoid the overhead of doing it in `evaluate` itself

        # The `evaluate` function will be called after every round
        def evaluate(
                server_round: int,
                parameters_ndarrays: fl.common.NDArrays,
                config: Dict[str, fl.common.Scalar],
        ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
                        
            # model path for saving local model
            gl_model_path = f'./{model_name}_gl_model_V{self.server.gl_model_v}'
            
            metrics = None
            
            if self.model_type == "Tensorflow":
                # loss, accuracy, precision, recall, auc, auprc = model.evaluate(x_val, y_val)
                loss, accuracy = model.evaluate(self.x_val, self.y_val)

                model.set_weights(parameters_ndarrays)  # Update model with the latest parameters
                
                # model save
                model.save(gl_model_path+'.tf')
            
            elif self.model_type == "Pytorch":
                import torch
                keys = [k for k in model.state_dict().keys() if "bn" not in k]
                params_dict = zip(keys, parameters_ndarrays)
                state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
                model.load_state_dict(state_dict, strict=True)
            
                loss, accuracy, metrics = self.test_torch(model, self.gl_val_loader, self.cfg)
                
                # model save
                torch.save(model.state_dict(), gl_model_path+'.pth')

            elif self.model_type == "Huggingface":
                logging.warning("Skipping evaluation for Huggingface model")
                loss, accuracy = 0.0, 0.0

                os.makedirs(gl_model_path, exist_ok=True)
                np.savez(os.path.join(gl_model_path, "adapter_parameters.npz"), *parameters_ndarrays)

            if self.server.round >= 1:
                # fit aggregation end time
                self.server.end_by_round = time.time() - self.server.start_by_round
                # gl model performance by round
                if metrics!=None:
                    server_eval_result = {"fl_task_id": self.task_id, "round": self.server.round, "gl_loss": loss, "gl_accuracy": accuracy,
                                      "run_time_by_round": self.server.end_by_round, **metrics,"gl_model_v":self.server.gl_model_v}
                else:
                    server_eval_result = {"fl_task_id": self.task_id, "round": self.server.round, "gl_loss": loss, "gl_accuracy": accuracy,
                                      "run_time_by_round": self.server.end_by_round,"gl_model_v":self.server.gl_model_v}
                json_server_eval = json.dumps(server_eval_result)
                logging.info(f'server_eval_result - {json_server_eval}')

                # send gl model evaluation to performance pod
                server_api.ServerAPI(self.task_id).put_gl_model_evaluation(json_server_eval)
                
            if metrics!=None:
                return loss, {"accuracy": accuracy, **metrics}
            else:
                return loss, {"accuracy": accuracy}

        return evaluate
    


    def fit_config(self, rnd: int):
        """Return training configuration dict for each round.
        Keep batch size fixed at 32, perform two rounds of training with one
        local epoch, increase to two local epochs afterwards.
        """
        fl_config = {
            "batch_size": self.batch_size,
            "local_epochs": self.local_epochs,
            "num_rounds": self.num_rounds,
        }

        # increase round
        self.server.round += 1

        # fit aggregation start time
        self.server.start_by_round = time.time()
        logging.info('server start by round')

        return fl_config


    def evaluate_config(self, rnd: int):
        """Return evaluation configuration dict for each round.
        """
        return {"batch_size": self.batch_size}


    def start(self):

        today_time = datetime.datetime.today().strftime('%Y-%m-%d %H-%M-%S')


        # Loaded last global model or no global model in s3
        self.next_model, self.next_model_name, self.server.last_gl_model_v = server_utils.model_download_s3(self.task_id, self.model_type, self.init_model)
        
        # Loaded last global model or no global model in local
        # self.next_model, self.next_model_name, self.server.latest_gl_model_v = server_utils.model_download_local(self.model_type, self.init_model)

        # logging.info('Loaded latest global model or no global model')

        # New Global Model Version
        self.server.gl_model_v = self.server.last_gl_model_v + 1

        # API that sends server status to server manager
        inform_Payload = {
            "S3_bucket": "fl-gl-model",  # bucket name
            "Last_GL_Model": "gl_model_%s_V.h5" % self.server.last_gl_model_v,  # Model Weight File Name
            "FLServer_start": today_time,
            "FLSeReady": True,  # server ready status
            "GL_Model_V": self.server.gl_model_v # Current Global Model Version
        }
        server_status_json = json.dumps(inform_Payload)
        server_api.ServerAPI(self.task_id).put_server_status(server_status_json)

        try:
            fl_start_time = time.time()

            # Run fl server
            gl_model_name = self.init_gl_model_registration(self.next_model, self.next_model_name)

            fl_end_time = time.time() - fl_start_time  # FL end time

            server_all_time_result = {"fl_task_id": self.task_id, "server_operation_time": fl_end_time,
                                      "gl_model_v": self.server.gl_model_v}
            json_all_time_result = json.dumps(server_all_time_result)
            logging.info(f'server_operation_time - {json_all_time_result}')
            
            # Send server time result to performance pod
            server_api.ServerAPI(self.task_id).put_server_time_result(json_all_time_result)
            
            # upload global model
            if self.model_type == "Tensorflow":
                global_model_file_name = f"{gl_model_name}_gl_model_V{self.server.gl_model_v}.h5"
                server_utils.upload_model_to_bucket(self.task_id, global_model_file_name)
            elif self.model_type =="Pytorch":
                global_model_file_name = f"{gl_model_name}_gl_model_V{self.server.gl_model_v}.pth"
                server_utils.upload_model_to_bucket(self.task_id, global_model_file_name)
            elif self.model_type == "Huggingface":
                global_model_file_name = f"{gl_model_name}_gl_model_V{self.server.gl_model_v}"
                npz_file_path = f"{global_model_file_name}.npz"
                # 경로 변경: evaluate에서 저장한 실제 경로
                # evaluate에서는: ./{gl_model_name}_gl_model_VN/adapter_parameters.npz 로 저장함
                model_dir = f"{global_model_file_name}"
                real_npz_path = os.path.join(model_dir, "adapter_parameters.npz")
                # 파일 이름 통일을 위해 복사 (선택)
                shutil.copy(real_npz_path, npz_file_path)

                server_utils.upload_model_to_bucket(self.task_id, npz_file_path)

            logging.info(f'upload {global_model_file_name} model in s3')

        # server_status error
        except Exception as e:
            logging.error('error: ', e)
            data_inform = {'FLSeReady': False}
            server_api.ServerAPI(self.task_id).put_server_status(json.dumps(data_inform))

        finally:
            logging.info('server close')

            # Modifying the model version in server manager
            server_api.ServerAPI(self.task_id).put_fl_round_fin()
            logging.info('global model version upgrade')
  ##############################
# GeneticFLServer 클래스 (Genetic CFL 기능 통합)
##############################
class GeneticFLServer:
    def __init__(self, cfg: Dict, model, model_name, model_type, gl_val_loader, test_torch, initial_hyperparams):
        self.cfg = cfg
        self.model = model
        self.model_name = model_name
        self.model_type = model_type
        self.gl_val_loader = gl_val_loader
        self.test_torch = test_torch
        # 구성 파일에 정의된 초기 후보 하이퍼파라미터 (예: [[0.001, 128], [0.005, 64], [0.01, 32]])
        self.hyperparams = initial_hyperparams  
        self.client_updates = []  # 클라이언트 업데이트 저장 리스트
        self.num_rounds = int(cfg.num_rounds)
        self.task_id = os.environ.get("TASK_ID", "default_task")
        self.server = server_utils.FLServerStatus()  # 서버 상태 객체

    def get_model_weights(self):
        return self.model.state_dict()
    
    def set_model_weights(self, new_state_dict):
        self.model.load_state_dict(new_state_dict)
    
    def send_to_clients(self, broadcast_data: Dict):
        # 실제 네트워크 전송 코드로 대체할 부분
        print("Broadcasting model and hyperparameters to clients:")
        print(broadcast_data)
    
    def collect_client_updates(self):
        """
        실제 클라이언트로부터 업데이트를 수집하는 함수.
        각 클라이언트는 { 'weights': state_dict, 'loss': 실제 손실 값, 'hyperparam': [learning_rate, batch_size] } 형태로 데이터를 전송합니다.
        이 함수는 실제 업데이트 수집 로직(예: 네트워크 통신, 메시지 큐 등)으로 대체되어야 합니다.
        """
        raise NotImplementedError("실제 클라이언트 업데이트 수집 로직을 구현하세요.")
    
    def aggregate_client_updates(self, client_updates):
        total_weights = None
        losses = []
        hyperparams_list = []  # 각 항목은 [lr, bs] 형태
        for update in client_updates:
            weights = update["weights"]
            loss = update["loss"]
            hyperparam = update["hyperparam"]  # 이미 [lr, bs] 형태
            losses.append(loss)
            hyperparams_list.append(hyperparam)
            if total_weights is None:
                total_weights = {k: v.clone() for k, v in weights.items()}
            else:
                for k in total_weights:
                    total_weights[k] += weights[k]
        for k in total_weights:
            total_weights[k] = total_weights[k] / len(client_updates)
            
        # DBSCAN 클러스터링: hyperparams_list -> numpy 배열 (shape=(n_samples, 2))
        hyperparams_array = np.array(hyperparams_list)
        scaled_array = hyperparams_array.copy()
        # 배치 크기는 로그 변환하여 스케일 보정
        scaled_array[:, 1] = np.log(scaled_array[:, 1])
        
        dbscan = DBSCAN(eps=0.1, min_samples=2)
        clusters = dbscan.fit_predict(scaled_array)
        print("DBSCAN clusters:", clusters)
        
        new_hyperparams = []
        unique_clusters = np.unique(clusters)
        for cluster_id in unique_clusters:
            if cluster_id == -1:
                continue
            indices = [i for i, c in enumerate(clusters) if c == cluster_id]
            cluster_losses = [losses[i] for i in indices]
            cluster_params = [hyperparams_list[i] for i in indices]
            evolved_cluster = evolve(cluster_losses, cluster_params)
            new_hyperparams.extend(evolved_cluster)
        if not new_hyperparams:
            new_hyperparams = self.hyperparams
        print("Evolved hyperparameters after clustering:", new_hyperparams)
        self.hyperparams = new_hyperparams
        
        # 서버 상태 전송: 클러스터링 결과를 API를 통해 전송
        status_payload = {
            "FL_task_id": self.task_id,
            "evolved_hyperparams": new_hyperparams,
            "num_client_updates": len(client_updates),
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        server_api.ServerAPI(self.task_id).put_server_status(json.dumps(status_payload))
        
        return total_weights
    
    def evaluate_global_model(self):
        loss, acc, metrics = self.test_torch(self.model, self.gl_val_loader, self.cfg)
        return loss, acc, metrics
    
    def broadcast_model_and_hyperparams(self):
        broadcast_data = {
            "model_weights": self.get_model_weights(),
            "hyperparams": self.hyperparams
        }
        self.send_to_clients(broadcast_data)
    
    def train_round(self, round_number: int):
        print(f"=== Starting Training Round {round_number} ===")
        self.broadcast_model_and_hyperparams()
        client_updates = self.collect_client_updates()  # 실제 클라이언트 업데이트 수집 로직 사용
        new_global_weights = self.aggregate_client_updates(client_updates)
        self.set_model_weights(new_global_weights)
        loss, acc, _ = self.evaluate_global_model()
        print(f"Round {round_number} - Global Model => Loss: {loss:.4f}, Accuracy: {acc:.4f}")
        eval_payload = {
            "round": round_number,
            "loss": loss,
            "accuracy": acc,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        server_api.ServerAPI(self.task_id).put_gl_model_evaluation(json.dumps(eval_payload))
    
    def register_and_upload_model(self):
        """
        FL 서버의 최종 글로벌 모델 등록 및 업로드 흐름.
        기존 FLServer의 흐름을 따라, 글로벌 모델 등록, 서버 운영 시간 측정,
        서버 시간 결과 전송, 그리고 S3 업로드를 수행합니다.
        """
        fl_start_time = time.time()
        gl_model_name = self.model_name  # 실제 등록 로직으로 대체 가능
        fl_end_time = time.time() - fl_start_time
        
        server_all_time_result = {
            "fl_task_id": self.task_id,
            "server_operation_time": fl_end_time,
            "gl_model_v": self.server.gl_model_v
        }
        json_all_time_result = json.dumps(server_all_time_result)
        logging.info(f'server_operation_time - {json_all_time_result}')
        server_api.ServerAPI(self.task_id).put_server_time_result(json_all_time_result)
        
        if self.model_type == "Tensorflow":
            global_model_file_name = f"{gl_model_name}_gl_model_V{self.server.gl_model_v}.h5"
            server_utils.upload_model_to_bucket(self.task_id, global_model_file_name)
        elif self.model_type == "Pytorch":
            global_model_file_name = f"{gl_model_name}_gl_model_V{self.server.gl_model_v}.pth"
            server_utils.upload_model_to_bucket(self.task_id, global_model_file_name)
        elif self.model_type == "Huggingface":
            global_model_file_name = f"{gl_model_name}_gl_model_V{self.server.gl_model_v}"
            npz_file_path = f"{global_model_file_name}.npz"
            model_dir = f"{global_model_file_name}"
            real_npz_path = os.path.join(model_dir, "adapter_parameters.npz")
            shutil.copy(real_npz_path, npz_file_path)
            server_utils.upload_model_to_bucket(self.task_id, npz_file_path)
        
        logging.info(f'upload {global_model_file_name} model in s3')
    
    def start(self):
        for r in range(1, self.num_rounds + 1):
            self.train_round(r)
        self.register_and_upload_model()
        final_payload = {"FLSeReady": False, "final_round": self.num_rounds}
        server_api.ServerAPI(self.task_id).put_server_status(json.dumps(final_payload))
        print("Global model training finished.")

##############################################
# server_main.py 진입점: GeneticFLServer 실행
##############################################
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

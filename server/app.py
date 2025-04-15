# server/app.py
import logging
from typing import Dict, Optional, Tuple
import flwr as fl
import datetime
import os
import json
import time
import numpy as np
import shutil
from collections import OrderedDict
from hydra.utils import instantiate
from sklearn.cluster import DBSCAN
from genetic_tuner import evolve  # 2차원 하이퍼파라미터용 유전 알고리즘 함수들
from . import server_api  
from . import server_utils 
from omegaconf import OmegaConf
# TF warning log filtering (필요 시)
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)8.8s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

##############################################
# 기존 FLServer 클래스 (필요 시 유지)
##############################################
class FLServer():
    def __init__(self, cfg, model, model_name, model_type, gl_val_loader=None, x_val=None, y_val=None, test_torch=None):
        self.task_id = os.environ.get('TASK_ID')  # FL Task ID
        self.server = server_utils.FLServerStatus()  # FLServerStatus class
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
        model_parameters = None
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
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=self.num_rounds),
            strategy=strategy,
        )

    def get_eval_fn(self, model, model_name):
        def evaluate(
                server_round: int,
                parameters_ndarrays: fl.common.NDArrays,
                config: Dict[str, fl.common.Scalar],
        ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
            gl_model_path = f'./{model_name}_gl_model_V{self.server.gl_model_v}'
            metrics = None
            if self.model_type == "Tensorflow":
                loss, accuracy = model.evaluate(self.x_val, self.y_val)
                model.set_weights(parameters_ndarrays)
                model.save(gl_model_path+'.tf')
            elif self.model_type == "Pytorch":
                import torch
                keys = [k for k in model.state_dict().keys() if "bn" not in k]
                params_dict = zip(keys, parameters_ndarrays)
                state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
                model.load_state_dict(state_dict, strict=True)
                loss, accuracy, metrics = self.test_torch(model, self.gl_val_loader, self.cfg)
                torch.save(model.state_dict(), gl_model_path+'.pth')
            elif self.model_type == "Huggingface":
                logging.warning("Skipping evaluation for Huggingface model")
                loss, accuracy = 0.0, 0.0
                os.makedirs(gl_model_path, exist_ok=True)
                np.savez(os.path.join(gl_model_path, "adapter_parameters.npz"), *parameters_ndarrays)
            if self.server.round >= 1:
                self.server.end_by_round = time.time() - self.server.start_by_round
                if metrics is not None:
                    server_eval_result = {"fl_task_id": self.task_id, "round": self.server.round, "gl_loss": loss, "gl_accuracy": accuracy,
                                          "run_time_by_round": self.server.end_by_round, **metrics, "gl_model_v": self.server.gl_model_v}
                else:
                    server_eval_result = {"fl_task_id": self.task_id, "round": self.server.round, "gl_loss": loss, "gl_accuracy": accuracy,
                                          "run_time_by_round": self.server.end_by_round, "gl_model_v": self.server.gl_model_v}
                json_server_eval = json.dumps(server_eval_result)
                logging.info(f'server_eval_result - {json_server_eval}')
                server_api.ServerAPI(self.task_id).put_gl_model_evaluation(json_server_eval)
            if metrics is not None:
                return loss, {"accuracy": accuracy, **metrics}
            else:
                return loss, {"accuracy": accuracy}
        return evaluate

    def fit_config(self, rnd: int):
        fl_config = {
            "batch_size": self.batch_size,
            "local_epochs": self.local_epochs,
            "num_rounds": self.num_rounds,
        }
        self.server.round += 1
        self.server.start_by_round = time.time()
        logging.info('server start by round')
        return fl_config

    def evaluate_config(self, rnd: int):
        return {"batch_size": self.batch_size}

    def start(self):
        today_time = datetime.datetime.today().strftime('%Y-%m-%d %H-%M-%S')
        self.next_model, self.next_model_name, self.server.last_gl_model_v = server_utils.model_download_s3(self.task_id, self.model_type, self.init_model)
        self.server.gl_model_v = self.server.last_gl_model_v + 1
        inform_Payload = {
            "S3_bucket": "fl-gl-model",
            "Last_GL_Model": "gl_model_%s_V.h5" % self.server.last_gl_model_v,
            "FLServer_start": today_time,
            "FLSeReady": True,
            "GL_Model_V": self.server.gl_model_v
        }
        server_status_json = json.dumps(inform_Payload)
        server_api.ServerAPI(self.task_id).put_server_status(server_status_json)
        try:
            fl_start_time = time.time()
            gl_model_name = self.init_gl_model_registration(self.next_model, self.next_model_name)
            fl_end_time = time.time() - fl_start_time
            server_all_time_result = {"fl_task_id": self.task_id, "server_operation_time": fl_end_time, "gl_model_v": self.server.gl_model_v}
            json_all_time_result = json.dumps(server_all_time_result)
            logging.info(f'server_operation_time - {json_all_time_result}')
            server_api.ServerAPI(self.task_id).put_server_time_result(json_all_time_result)
            if self.model_type == "Tensorflow":
                global_model_file_name = f"{gl_model_name}_gl_model_V{self.server.gl_model_v}.h5"
                server_utils.upload_model_to_bucket(self.task_id, global_model_file_name)
            elif self.model_type =="Pytorch":
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
        except Exception as e:
            logging.error('error: ', e)
            data_inform = {'FLSeReady': False}
            server_api.ServerAPI(self.task_id).put_server_status(json.dumps(data_inform))
        finally:
            logging.info('server close')
            server_api.ServerAPI(self.task_id).put_fl_round_fin()
            logging.info('global model version upgrade')

##############################################
# GeneticFLServer 클래스 (Genetic CFL 기능 통합)
##############################################
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
        self.server = server_utils.FLServerStatus()  # 글로벌 모델 버전 등 상태 관리 객체

    def get_model_weights(self):
        return self.model.state_dict()
    
    def set_model_weights(self, new_state_dict):
        self.model.load_state_dict(new_state_dict)
    
    def send_to_clients(self, broadcast_data: Dict):
        # 실제 네트워크 전송 대신 출력
        print("Broadcasting model and hyperparameters to clients:")
        print(broadcast_data)
    
    def collect_client_updates(self):
        """
        실제 클라이언트로부터 업데이트를 수집하는 함수.
        각 클라이언트는 { 'weights': state_dict, 'loss': 실제 손실 값, 'hyperparam': [learning_rate, batch_size] }
        형태로 데이터를 전송해야 합니다.
        실제 업데이트 수집 로직(예: 네트워크 통신, 메시지 큐 등)으로 대체되어야 하며,
        여기서는 예시용 dummy update를 반환합니다.
        """
        dummy_update = {
            "weights": self.get_model_weights(),
            "loss": 0.25,  # 실제 손실 값이 들어가야 합니다.
            "hyperparam": self.hyperparams[0]  # 예: [0.001, 128]
        }
        print("Collecting client updates (dummy update)...")
        return [dummy_update]
    
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
            
        # DBSCAN 클러스터링: hyperparams_list → numpy 배열 (n_samples, 2)
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
        
        from omegaconf import OmegaConf
        status_payload = {
            "FL_task_id": self.task_id,
            "evolved_hyperparams": self.hyperparams,
            "num_client_updates": len(client_updates),
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        status_payload_native = OmegaConf.to_container(OmegaConf.create(status_payload), resolve=True)
        server_api.ServerAPI(self.task_id).put_server_status(json.dumps(status_payload_native))
        
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
        client_updates = self.collect_client_updates()
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
        FL 서버의 최종 글로벌 모델 등록 및 업로드 과정.
        기존 FLServer의 흐름을 따라, 글로벌 모델 등록, 서버 운영 시간 측정,
        서버 시간 결과 전송, 그리고 S3 업로드를 수행합니다.
        """
        fl_start_time = time.time()
        gl_model_name = self.model_name  # 실제 등록 로직에 따라 수정 가능
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
        elif self.model_type == "Pytorch":
            global_model_file_name = f"{gl_model_name}_gl_model_V{self.server.gl_model_v}.pth"
        elif self.model_type == "Huggingface":
            global_model_file_name = f"{gl_model_name}_gl_model_V{self.server.gl_model_v}"
        else:
            logging.error("Unknown model type for upload.")
            return
        
        file_path = f"./{global_model_file_name}"
        if not os.path.exists(file_path):
            logging.error(f"Global model file not found: {file_path}")
            raise FileNotFoundError(f"Global model file not found: {file_path}")
        
        try:
            server_utils.upload_model_to_bucket(self.task_id, global_model_file_name)
            logging.info(f'upload {global_model_file_name} model in s3')
        except Exception as upload_err:
            logging.error("Error during S3 upload:", exc_info=upload_err)
            raise upload_err
    
    def start(self):
        for r in range(1, self.num_rounds + 1):
            self.train_round(r)
        self.register_and_upload_model()
        final_payload = {"FLSeReady": False, "final_round": self.num_rounds}
        server_api.ServerAPI(self.task_id).put_server_status(json.dumps(final_payload))
        print("Global model training finished.")

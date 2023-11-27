import torch
from datetime import datetime
import os

# HYPERPARAMETERS
batch_size = 128
epochs = 10
lr = 0.001
ITERATION = 11
alpha = 0.1
pca_components = 23

# DIRECTORY PATHS
MODEL_DIR = "weights"
MODEL_PATH = f"model/{MODEL_DIR}/autoencoder_weights.pth"
OUTPUT_DIR = "results"

# 현재 날짜와 시간을 가져와서 문자열 형식으로 변환
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# 하위 폴더 이름 설정
subfolder_name = f"run_{current_time}"

# 최종 경로 생성
SUBFOLDER_PATH = os.path.join(OUTPUT_DIR, subfolder_name)

# 폴더가 존재하지 않으면 생성
if not os.path.exists(SUBFOLDER_PATH):
    os.makedirs(SUBFOLDER_PATH)

# 결과 파일 경로
losses_file_path = "average_losses.txt"
LOSSES_PATH = os.path.join(SUBFOLDER_PATH, losses_file_path)

# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

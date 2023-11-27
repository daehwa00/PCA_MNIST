import torch
import os
from model import ConvAutoencoder
from utils import load_data, train, process_and_save_images, save_losses_to_file
from config import *


def main():
    # 데이터 로딩
    train_loader = load_data(batch_size)

    # 모델 초기화
    model = ConvAutoencoder().to(device)

    # 손실 함수 및 옵티마이저 정의
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Autoencoder 훈련
    if not os.path.exists(MODEL_PATH):
        train(model, criterion, optimizer, train_loader, epochs, device)
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Saved model weights to {MODEL_PATH}")
    else:
        model.load_state_dict(torch.load(MODEL_PATH))
        print(f"Loaded model weights from {MODEL_PATH}")

    # 이미지 변형 및 저장
    model.eval()
    for steps in range(1, ITERATION):
        process_and_save_images(model, train_loader, steps)

    # 평균 손실 저장
    save_losses_to_file(losses_file_path)


if __name__ == "__main__":
    main()

import torch
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
from sklearn.decomposition import PCA
from config import *

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)


def load_data(batch_size):
    # 데이터 전처리 및 로딩
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    return train_loader


def train(model, criterion, optimizer, train_loader, epochs):
    # 모델 훈련 함수
    model.train()
    for epoch in range(epochs):
        for data in train_loader:
            inputs, _ = data
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")


def process_and_save_images(model, train_loader, steps):
    """
    모든 이미지에 대해 PCA를 수행하고, 변형된 이미지와 레이블을 튜플로 저장합니다.
    """
    encoded_images = []  # 이미지를 인코딩한 벡터를 저장할 리스트
    transformed_data = []  # 이미지와 레이블을 튜플로 저장할 리스트
    losses = []
    labels = []
    mean_vectors = []

    with torch.no_grad():
        for images, lbls in train_loader:
            encoded = model.encoder(images.to(device))
            encoded_images.extend(encoded.cpu().numpy().reshape(len(images), -1))
            labels.extend(lbls.cpu().numpy())

    encoded_images = np.array(encoded_images)
    labels = np.array(labels)

    pca = PCA(n_components=pca_components)
    pca.fit(encoded_images)

    for i in range(10):
        label_indices = np.where(labels == i)[0]
        label_vectors = pca.transform(encoded_images[label_indices])
        mean_vector = np.mean(label_vectors, axis=0)
        mean_vectors.append(mean_vector)

    for idx, (image, label) in enumerate(
        tqdm(train_dataset, desc=f"Processing num_steps={steps}")
    ):
        transformed_img, img_loss = transform_image(
            model, image.unsqueeze(0), label, steps, pca, mean_vectors
        )
        transformed_data.append(
            (transformed_img.squeeze().numpy(), label)
        )  # 이미지와 레이블을 튜플로 추가
        losses.append(img_loss)

    numpy_transformed_images = np.array([data[0] for data in transformed_data])
    transformed_images_tensor = torch.tensor(numpy_transformed_images)

    labels_tensor = torch.tensor([data[1] for data in transformed_data])
    transformed_dataset = {"images": transformed_images_tensor, "labels": labels_tensor}
    transformed_dataset_path = f"transformed_dataset_{steps}.pth"
    OUTPUT_PATH = f"{OUTPUT_DIR}/{transformed_dataset_path}"

    torch.save(transformed_dataset, OUTPUT_PATH)  # 딕셔너리 형태로 저장
    print(f"Saved transformed images and labels to {OUTPUT_PATH}")

    # 모든 이미지에 대한 loss 평균 출력 및 파일에 저장
    average_loss = np.mean(losses)
    save_losses_to_file(LOSSES_PATH, steps, average_loss)


def transform_image(model, image, label, num_steps, pca, mean_vectors):
    image.requires_grad_(True)
    optimizer = torch.optim.SGD([image], lr=alpha)

    for step in range(num_steps):
        optimizer.zero_grad()
        encoded = model.encoder(image.to(device)).view(1, -1)
        pca_encoded = pca.transform(encoded.detach().cpu().numpy())

        # 다른 라벨의 평균 벡터들 중 L1 노름이 가장 작은 벡터 선택
        l1_norms = np.linalg.norm(mean_vectors - pca_encoded, ord=1, axis=1)
        l1_norms[label] = np.inf  # 현재 라벨은 제외
        closest_mean_vector_idx = np.argmin(l1_norms)
        closest_mean_vector = mean_vectors[closest_mean_vector_idx]

        # PCA 역변환을 통해 원래 차원으로 복원
        closest_mean_vector_original_dim = pca.inverse_transform([closest_mean_vector])

        # L2 노름을 손실로 설정
        loss = torch.norm(
            encoded
            - torch.tensor(
                closest_mean_vector_original_dim, device=device, dtype=encoded.dtype
            ),
            p=2,
        )
        loss.backward()

        # 이미지 픽셀 값 업데이트
        optimizer.step()

        # 이미지 값이 [0, 1] 범위를 벗어나지 않도록 클리핑
        with torch.no_grad():
            image.clamp_(0, 1)

    return image.detach(), loss.item()


def save_losses_to_file(LOSSES_PATH, num_steps, average_loss):
    with open(LOSSES_PATH, "a") as file:
        file.write(f"Average Loss for {num_steps} steps: {average_loss:.4f}\n")

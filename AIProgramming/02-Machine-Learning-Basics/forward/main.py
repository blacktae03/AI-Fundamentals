import numpy as np
import matplotlib.pyplot as plt
from dataset import load_data
from dataloader import DataLoader
from model import MyModel
from train import train


def main():
    """
    Iris 분류 모델 학습 메인 함수
    - 데이터 로드 → DataLoader 구성 → 모델 생성 → 학습 루프 실행
    - 각 epoch의 평균 손실(loss history)을 시각화
    """
    # 1. 데이터 로드
    dataset = load_data("./before_final/hw006/iris.csv")

    # 2. DataLoader 생성
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # 3. 모델 초기화
    model = MyModel()

    # 4. 훈련 수행 (평균 손실 기록 반환)
    loss_history = train(model, dataloader, num_epochs=10)

    # 5. 손실 히스토리 시각화
    if len(loss_history) == 0:
        print("⚠️ Loss history is empty. Check the train() function output.")
        return

    epochs = np.arange(1, len(loss_history) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, loss_history, marker='o', color='blue', label='Mean Loss per Epoch')
    plt.title("Iris Model Training Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Loss Value")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

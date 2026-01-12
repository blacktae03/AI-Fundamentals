from sklearn.datasets import load_digits
import numpy as np
from knn import MyKNNClassifier

def my_train_test_split(X, y, test_size=0.3, random_seed=777):
    """
    데이터를 학습용과 테스트용으로 분할합니다.
    재현성을 위해 random_seed를 고정합니다.
    """
    np.random.seed(random_seed)
    
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)

    # 비복원 추출로 테스트 인덱스 선택
    test_idx = np.random.choice(n_samples, size=n_test, replace=False)
    
    # 나머지 인덱스를 학습용으로 사용
    train_idx = np.setdiff1d(np.arange(n_samples), test_idx)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    return X_train, X_test, y_train, y_test

def main():
    # 1. 데이터 로드 및 분할
    digits = load_digits()
    X = digits.data
    y = digits.target
    
    # 시드를 고정하여 항상 같은 데이터셋으로 테스트하도록 함
    X_train, X_test, y_train, y_test = my_train_test_split(X, y, test_size=0.3, random_seed=777)

    #print(f"Test Set Shape: {X_test.shape}")
    #print("-" * 30)

    # 2. 유클리드 거리 테스트
    #print("Testing Euclidean Distance...")
    knn = MyKNNClassifier(k=3, distance_metric='e')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    #print(f" -> Accuracy: {accuracy:.4f}")

    # 3. 맨해튼 거리 테스트
    #print("Testing Manhattan Distance...")
    knn_man = MyKNNClassifier(k=3, distance_metric='m')
    knn_man.fit(X_train, y_train)
    y_pred_man = knn_man.predict(X_test)
    accuracy_man = np.mean(y_pred_man == y_test)
    #print(f" -> Accuracy: {accuracy_man:.4f}")

    #print("-" * 30)

    # 4. 최종 결과 판정
    if accuracy > 0.9 and accuracy_man > 0.9:
        print("PASS")
    else:
        print("FAIL")
        print("Hint: 정확도가 0.9(90%) 미만입니다.")
        print("거리 계산 수식이나, k-최근접 이웃 선택 로직(argsort 등)을 다시 확인해보세요.")

if __name__ == "__main__":
    main()
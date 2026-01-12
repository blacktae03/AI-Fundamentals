from abc import ABC, abstractmethod
import numpy as np


class Learning(ABC):
    """
    학습기 기본 뼈대를 제공하는 추상 클래스입니다.
    모든 학습 알고리즘은 이 클래스를 상속받아 fit과 predict 메서드를 구현해야 합니다.
    """

    @abstractmethod
    def fit(self, x, y):
        """
        학습 데이터를 저장합니다.

        Args:
            x: 학습용 특성 데이터
            y: 학습용 레이블 데이터
        """
        pass

    @abstractmethod
    def predict(self, x):
        """
        입력된 테스트 데이터에 대해 예측 값을 반환합니다.

        Args:
            x: 테스트용 특성 데이터

        Returns:
            예측된 레이블 배열
        """
        pass


class MyKNNClassifier(Learning):
    """
    K-최근접 이웃(K-Nearest Neighbors) 분류기를 구현한 클래스입니다.
    주어진 k개의 가장 가까운 이웃을 찾아 다수결 투표로 클래스를 예측합니다.
    """

    def __init__(self, k, distance_metric='e'):
        """
        KNN 분류기를 초기화합니다.

        Args:
            k (int): 고려할 최근접 이웃의 개수
            distance_metric (str): 거리 측정 방식
                - 'e': 유클리드 거리 (Euclidean Distance)
                - 'm': 맨해튼 거리 (Manhattan Distance)
        """
        self.k = k
        self.distance_metric = distance_metric


    def fit(self, x, y):
        """
        학습 데이터를 저장합니다.
        KNN은 게으른 학습(lazy learning) 방식이므로,
        학습 단계에서는 데이터를 저장만 합니다.

        Args:
            x (np.ndarray): 학습용 특성 데이터 (n_samples, n_features)
            y (np.ndarray): 학습용 레이블 데이터 (n_samples,)
        """

        self.x = x
        self.y = y


    def _calculate_distances(self, X_test):
        if self.distance_metric == 'e':
            # return np.sum(X_test[:, np.newaxis, :] - self.x[np.newaxis])
            X_test_ss = np.sum(X_test**2, axis=1, keepdims=True)
            x_ss = np.sum(self.x**2, axis=1)
            return np.sqrt(np.maximum(X_test_ss + x_ss - 2*np.dot(X_test, self.x.T), 0))
        
        else:
            # 맨해튼 거리는 브로드캐스팅 방식이 맞습니다.
            return np.sum(np.abs(X_test[:, np.newaxis, :] - self.x[np.newaxis, :, :]), axis=-1)

    def predict(self, x):
        sorted_indices = np.argsort(self._calculate_distances(x))
        k_sorted_indices = sorted_indices[:, :self.k]
        k_sorted_labels = self.y[k_sorted_indices]
        pred_ans = [np.argmax(np.bincount(labels)) for labels in k_sorted_labels]

        return np.array(pred_ans)

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
        pass

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
        pass

    def _calculate_distances(self, X_test):
        """
        테스트 데이터와 모든 학습 데이터 사이의 거리를 계산합니다.

        힌트:
            - 유클리드 거리는 수식 전개 방식을 사용하세요: ||a - b||² = ||a||² + ||b||² - 2a·b
            - 실습 노트북의 pairwise_distance 함수 구현을 참고하세요
            - np.sum(..., axis=1, keepdims=True)로 각 샘플의 제곱합 계산
            - np.dot(X_test, X_train.T)로 내적 계산
            - 부동소수점 오차 방지: np.maximum(..., 0)으로 음수 방지
            - 맨해튼 거리: sum(|x - y|) - 브로드캐스팅 사용

        Args:
            X_test (np.ndarray): 테스트 데이터 (n_test, n_features)

        Returns:
            np.ndarray: 거리 행렬 (n_test, n_train)
                각 (i, j) 원소는 test[i]와 train[j] 사이의 거리
        """
        diff = X_test[:, np.newaxis, :] - self.x[np.newaxis, :, :]
        
        if self.distance_metric == 'e': return np.sum(np.square(diff), axis=-1)
        else : return np.sum(np.abs(diff), axis=-1)
        pass

    def predict(self, x):
        """
        입력된 테스트 데이터에 대해 예측 클래스를 반환합니다.

        각 테스트 데이터에 대해:
        1. 모든 학습 데이터와의 거리를 계산
        2. 거리가 가까운 k개의 이웃을 선택
        3. k개 이웃의 레이블 중 가장 빈번한 클래스를 예측 결과로 반환

        힌트:
            - np.argsort를 사용하여 거리가 가까운 순서대로 인덱스 정렬
            - np.bincount로 각 레이블의 등장 횟수 계산
            - np.argmax로 가장 많이 등장한 레이블 찾기

        Args:
            x (np.ndarray): 테스트 데이터 (n_test, n_features)

        Returns:
            np.ndarray: 예측된 레이블 배열 (n_test,)
        """
        knn_indices = np.argsort(self._calculate_distances(x), axis=1)[:, :self.k] # (n_test, k)
        # 몇 번째 결과값이 제일 거리가 짧은지를 구했으면, 그것의 정답라벨을 저장해야함.
        knn_labels = self.y[knn_indices]
        # 그래야 여기서 정답 라벨 중에 제일 많이 나온게 뭐냐? 해서 답을 정하는 거임.
        ret = []
        for arr in knn_labels :
            ret.append((np.bincount(arr)).argmax())

        return np.array(ret)
        pass

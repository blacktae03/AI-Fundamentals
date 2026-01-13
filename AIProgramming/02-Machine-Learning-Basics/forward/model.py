import numpy as np

class Dense:
    """
    완전 연결(Dense) 계층 클래스
    - 입력과 가중치의 행렬 곱 연산 + 편향 추가 수행
    - 가중치는 He 초기화(He Initialization) 방식으로 설정됩니다.
    """
    def __init__(self, input_dim, output_dim):
        """
        Dense 계층의 가중치(W)와 편향(b)을 초기화합니다.
        He 초기화: W ~ N(0, sqrt(2 / input_dim))
        W: (input_dim, output_dim)
        b: (1, output_dim)
        """
        He = np.sqrt(2 / input_dim)
        self.W = np.random.normal(0, He, (input_dim, output_dim))
        self.b = np.zeros((1, output_dim))
        pass

    def forward(self, X):
        """
        입력 X에 대해 선형 변환을 수행합니다.
        Z = XW + b
        """
        return X @ self.W + self.b
        pass

    def __call__(self, X):
        """
        layer(X) 형태로 호출 시 forward()를 자동 실행합니다.
        """
        return self.forward(X)
        pass
    
class ReLU:
    """
    ReLU (Rectified Linear Unit) 활성화 함수 클래스
    - forward(): 입력에 대해 ReLU 연산 수행
    """
    def __init__(self):
        """ReLU에 필요한 내부 변수를 초기화합니다."""
        self.mask = None
        pass

    def forward(self, Z):
        """입력 Z에서 0보다 큰 값만 통과시키는 연산을 수행합니다."""
        self.mask = Z > 0
        return np.maximum(0, Z)
        pass
    
    def __call__(self, Z):
        """
        layer(Z) 형태로 호출 시 forward()를 자동 실행합니다.
        """
        return self.forward(Z)
        pass
    

class Softmax:
    """
    Softmax 활성화 함수 클래스
    - forward(): 입력 로짓을 확률 분포로 변환
    """
    def __init__(self):
        """Softmax에 필요한 내부 변수를 초기화합니다."""
        self.output = None
        pass

    def forward(self, Z):
        """입력 Z를 확률 벡터로 변환합니다. (수치 안정성 포함)"""
        bunmo = np.sum(np.exp(Z - np.max(Z, axis=1, keepdims=True)), axis=1, keepdims=True)
        self.output = np.exp(Z - np.max(Z, axis=1, keepdims=True)) / bunmo

        return self.output
        pass

    def __call__(self, Z):
        """
        layer(Z) 형태로 호출 시 forward()를 자동 실행합니다.
        """
        return self.forward(Z)
        pass

class MyModel:
    """
    Iris 분류를 위한 2층 완전 연결 신경망 (Dense → ReLU → Dense → Softmax)
    """
    def __init__(self):
        """가중치(W₁, W₂)와 편향(b₁, b₂)을 초기화합니다."""
        self.dense1 = Dense(4, 10)
        self.relu = ReLU()
        self.dense2 = Dense(10, 3)
        self.softmax = Softmax()
        pass

    def forward(self, X):
        """입력 X를 순전파(Forward Pass)하여 각 클래스의 확률을 반환합니다."""
        Z1 = self.dense1(X)
        A1 = self.relu(Z1)
        Z2 = self.dense2(A1)
        A2 = self.softmax(Z2)

        return A2
        pass

    def __call__(self, X):
        """model(X) 형태로 호출 시 forward()를 자동 실행합니다."""
        return self.forward(X)
        pass

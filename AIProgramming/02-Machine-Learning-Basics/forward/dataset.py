import numpy as np

class Dataset:
    """
    Iris 데이터셋을 관리하는 클래스
    - NumPy 배열로 구성된 X(특성), y(레이블)를 저장 및 접근
    """

    def __init__(self, X, y):
        """X와 y를 멤버 변수로 저장하고, 샘플 수를 검증합니다."""
        if X.shape[0] != y.shape[0]:
            raise ValueError("X와 y의 샘플 개수가 일치하지 않습니다.")
        
        self.X = X
        self.y = y
        pass

    def __len__(self):
        """데이터셋의 총 샘플 수를 반환합니다."""
        return len(self.X)
        pass

    def __getitem__(self, idx):
        """인덱스에 해당하는 하나의 (X, y) 샘플을 반환합니다."""
        return (self.X[idx], self.y[idx])
        pass


def load_data(filepath):
    """
    iris.csv 파일을 읽고 Dataset 인스턴스를 생성합니다.
    1. numpy.genfromtxt로 CSV 로드
    2. 문자열 레이블을 원-핫 인코딩
    3. Dataset(X, y) 반환
    """

    label_map = {'Setosa': 0, 'Versicolor': 1, 'Virginica': 2}


    data = np.genfromtxt(filepath, dtype=str, delimiter=',', skip_header=1)
    X = data[:, :4]
    y = data[:, 4]
    X = X.astype(float)
    y = np.char.strip(y, '"')
    vectorizer = np.vectorize(label_map.get)
    y = vectorizer(y).astype(int)
    y_one_hot = np.eye(3)[y]

    return Dataset(X, y_one_hot)
    pass

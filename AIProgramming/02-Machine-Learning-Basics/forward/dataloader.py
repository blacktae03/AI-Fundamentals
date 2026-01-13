import math
import random
import numpy as np

class DataLoader:
    """
    Dataset 객체를 받아서 데이터를 미니배치로 나누어 제공하는 클래스
    - shuffle 여부에 따라 데이터 순서를 섞을 수 있음
    """

    def __init__(self, dataset, batch_size=8, shuffle=True):
        """Dataset 객체와 배치 크기, 셔플 옵션을 초기화합니다."""
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.n_samples = len(dataset)

        self.indices = list(range(self.n_samples))
        pass

    def __iter__(self):
        """이터레이터 초기화: epoch 시작 시 데이터 인덱스를 셔플합니다."""
        if self.shuffle : np.random.shuffle(self.indices)
        self.current_idx = 0
        return self
        pass

    def __next__(self):
        """다음 배치를 반환합니다. 데이터가 끝나면 StopIteration을 발생시킵니다."""
        if self.current_idx >= len(self.dataset):
            raise StopIteration
        
        batch_indices = self.indices[self.current_idx : self.current_idx + self.batch_size]
        batch_x = self.dataset.X[batch_indices]
        batch_y = self.dataset.y[batch_indices]
        self.current_idx += self.batch_size
        return batch_x, batch_y
        pass

    def __len__(self):
        """한 epoch 동안 생성되는 총 배치 수를 반환합니다."""
        return math.ceil(self.n_samples / self.batch_size)
        pass

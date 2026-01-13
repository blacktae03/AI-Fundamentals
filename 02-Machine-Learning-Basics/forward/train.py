import numpy as np

def loss_function(predictions, targets):
    """
    단순 차이를 기반으로 한 손실 함수
    """
    return float(np.mean(np.square(predictions - targets)))


def train(model, dataloader, num_epochs=1):
    """
    모델을 학습시키는 루프 함수
    - 각 배치별로 순전파 수행
    - 손실 계산 및 평균 손실 기록
    - 각 epoch별 평균 손실 목록을 반환
    """
    loss_history = []
    for n in range(num_epochs) :
        batch_losses = []
        for x, y in dataloader:
            y_pred = model(x)
            
            loss = loss_function(y_pred, y)
            batch_losses.append(loss)

        loss_history.append(np.mean(batch_losses))
        
    return loss_history
    pass

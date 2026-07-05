
class EarlyStopping:
    def __init__(self, patience=10, mode='min', delta=0.001):
        """
        Args:
            patience: 容忍的指标不再改善的epoch数
            mode: 'min'表示监控指标越小越好（如loss），'max'表示越大越好（如F1）
            delta: 认为指标有改善的最小变化量
        """
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
        elif (self.mode == 'min' and current_score < self.best_score - self.delta) or \
             (self.mode == 'max' and current_score > self.best_score + self.delta):
            self.best_score = current_score
            self.counter = 0  # 重置计数器
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop
from ..base import BaseEstimator
import numpy as np


class MixedEffectsModel(BaseEstimator):
    def __init__(self, group_labels):
        super().__init__()
        self.group_labels = np.asarray(group_labels)
        self.random_effects_ = None

    def _solve(self, X, y):
        # 단순한 random intercept model로 시작
        # y = X * beta + Z * u + e
        # 여기서 Z는 group 구조
        # 단순히 그룹 평균 보정하는 정도로 시작하고, 나중에 EM 추가 가능
        pass

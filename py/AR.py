import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from tqdm.notebook import tqdm

class ARModel:
    def __init__(self, order: int = 1, alpha: float = 1.0) -> None:
        """
        Initializes an instance of the ARModel class.

        Args:
        - order (int): The order of the autoregressive model. Default is 1.
        - alpha (float): The regularization strength of the Ridge regression. Default is 1.0.
        """
        self.order = order
        self.alpha = alpha
        self.model = Ridge(alpha=self.alpha, solver='cholesky')
        self.score = 0

    def fit(self, x) -> None:
        """
        Fits the autoregressive model to the input data.

        Args:
        - X (np.ndarray): 1D numpy array of shape (n_samples,)

        Returns:
        - None
        """
        n_samples = len(x) - self.order
        X = np.zeros((n_samples, self.order))
        y = x[self.order:]
        for i in range(self.order):
            X[:, i] = x[i:i+n_samples]
        self.model.fit(X, y)
        result = self.model.predict(X)
        self.score = np.mean((result - y)**2)
        plt.plot(np.concatenate((x[:self.order+1], result)), label="prediction")

    def gridSearch_alpha(self, alphas, x) -> None:
        n_samples = len(x) - self.order
        X = np.zeros((n_samples, self.order))
        y = x[self.order:]
        for i in range(self.order):
            X[:, i] = x[i:i+n_samples]
            
        scores = [- cross_val_score(Ridge(alpha=alpha, solver='cholesky'), X, y, cv=5, scoring = "neg_mean_squared_error", n_jobs=1).mean() for alpha in alphas]
        
        plt.plot(alphas, scores)
        print(alphas[np.argmin(scores)])
        
    def gridSearch_order(self, orders, x):
        scores = []
        for order in tqdm(orders):
            n_samples = len(x) - order
            X = np.zeros((n_samples, order))
            y = x[order:]
            for i in range(order):
                X[:, i] = x[i:i+n_samples]
            scores.append(-cross_val_score(Ridge(alpha=self.alpha, solver='cholesky'), X, y, cv=5, scoring = "neg_mean_squared_error", n_jobs=1).mean())
        
        plt.plot(orders, scores)
        print(orders[np.argmin(scores)])
            
        
        
        
        
        
        

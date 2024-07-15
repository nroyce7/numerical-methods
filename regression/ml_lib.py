import sys
import numpy as np
import matplotlib.pyplot as plt


class SimpleLinear(object):
    def __init__(self, x, y):
        self.x = np.copy(x)
        self.y = np.copy(y)
        x_avg = np.average(x)
        y_avg = np.average(y)
        self.beta = np.zeros(2)
        self.beta[1] = np.dot(x - x_avg, y - y_avg) / np.dot(x - x_avg, x - x_avg)
        self.beta[0] = y_avg - self.beta[1] * x_avg
        self.RMSE_train = np.sqrt(np.average((y - (self.beta[1] * x + self.beta[0])) ** 2.0))
        self.R2_train = 1.0 - self.RMSE_train ** 2.0 / np.average((y - y_avg) ** 2.0)

    def predict(self, x):
        return self.beta[1] * x + self.beta[0]

    def RMSE(self, x=None, y=None):
        if x is None or y is None:
            return self.RMSE_train
        return np.sqrt(np.average((self.predict(x) - y) ** 2.0))

    def R2(self, x=None, y=None):
        if x is None or y is None:
            return self.R2_train
        return 1.0 - self.RMSE(x, y) ** 2.0 / np.average((y - np.average(y)) ** 2.0)

    def plot(self, x=None, y=None):
        if x is None or y is None:
            x = self.x
            y = self.y
        x_values = np.linspace(np.min(x), np.max(x), num=2)
        plt.scatter(x, y, c='darkblue', s=2.0)
        plt.plot(x_values, self.predict(x_values), c='r')
        plt.show()


class MultipleLinear(object):
    def __init__(self, X, y):
        design_matrix = np.append(np.ones((X.shape[0], 1)), X, axis=-1)
        self.cov = design_matrix.T @ design_matrix
        self.beta = np.linalg.inv(self.cov) @ design_matrix.T @ y
        self.RMSE_train = np.sqrt(np.average((y - design_matrix @ self.beta) ** 2))
        self.R2_train = 1.0 - self.RMSE_train ** 2.0 / np.average((y - np.average(y)) ** 2.0)

    def predict(self, X):
        return X @ self.beta[1:] + self.beta[0]

    def RMSE(self, X=None, y=None):
        if X is None or y is None:
            return self.RMSE_train
        return np.sqrt(np.average((self.predict(X) - y) ** 2.0))

    def R2(self, X=None, y=None):
        if X is None or y is None:
            return self.R2_train
        return 1.0 - self.RMSE(X, y) ** 2.0 / np.average((y - np.average(y)) ** 2.0)


class Ridge(object):
    def __init__(self, X, y, alpha=1.0):
        design_matrix = np.append(np.ones((X.shape[0], 1)), X, axis=-1)
        self.cov = design_matrix.T @ design_matrix
        regularization = alpha * np.identity(self.cov.shape[0])
        regularization[0, 0] = 0.0
        self.beta = np.linalg.inv(self.cov + regularization) @ design_matrix.T @ y
        self.RMSE_train = np.sqrt(np.average((y - design_matrix @ self.beta) ** 2))
        self.R2_train = 1.0 - self.RMSE_train ** 2.0 / np.average((y - np.average(y)) ** 2.0)

    def predict(self, X):
        return X @ self.beta[1:] + self.beta[0]

    def RMSE(self, X=None, y=None):
        if X is None or y is None:
            return self.RMSE_train
        return np.sqrt(np.average((self.predict(X) - y) ** 2.0))

    def R2(self, X=None, y=None):
        if X is None or y is None:
            return self.R2_train
        return 1.0 - self.RMSE(X, y) ** 2.0 / np.average((y - np.average(y)) ** 2.0)


class Lasso(object):
    def __init__(self, X, y, alpha=1.0, TOL=1e-3, MAX=500):
        # Center the data
        X_average = np.average(X, axis=0)
        X_centered = X - X_average
        y_average = np.average(y)
        y_centered = y - y_average

        self.cov = np.array(X_centered.T @ X_centered)      # Casting this to a numpy array prevents dataframe input from crashing
        design_at_y = X_centered.T @ y_centered

        # Use Ridge with alpha = 1.0 for a starting guess
        self.beta = np.linalg.inv(self.cov + np.identity(self.cov.shape[0])) @ design_at_y

        # Precompute values that will be used later
        self.cov /= X.shape[0]
        design_at_y /= X.shape[0]
        x_var = np.average(X_centered ** 2.0, axis=0)

        old_beta = np.zeros(self.beta.shape)
        counter = 0

        # Loop through coordinate descent
        while np.max(np.abs(old_beta - self.beta)) > TOL and counter < MAX:
            old_beta = np.copy(self.beta)
            for i in range(self.beta.shape[0]):
                self.beta[i] = 0
                partial_beta_i = self.cov[i, :] @ self.beta - design_at_y[i]
                if partial_beta_i < -alpha:
                    self.beta[i] = -(partial_beta_i + alpha) / x_var[i]
                elif partial_beta_i > alpha:
                    self.beta[i] = -(partial_beta_i - alpha) / x_var[i]
                else:
                    self.beta[i] = 0.0
            # If within TOL of 0 just reduce to 0
            self.beta[np.abs(self.beta) < TOL] = 0.0
            counter += 1

        if np.max(np.abs(old_beta - self.beta)) > TOL:
            print("Failed to converge.", file=sys.stderr)

        # Add in beta_0 term since the data was shifted to have mean 0
        self.beta = np.insert(self.beta, 0, y_average - X_average.T @ self.beta, axis=0)

        self.RMSE_train = np.sqrt(np.average((y - X @ self.beta[1:] - self.beta[0]) ** 2))
        self.R2_train = 1.0 - self.RMSE_train ** 2.0 / np.average((y - np.average(y)) ** 2.0)

    def predict(self, X):
        return X @ self.beta[1:] + self.beta[0]

    def RMSE(self, X=None, y=None):
        if X is None or y is None:
            return self.RMSE_train
        return np.sqrt(np.average((self.predict(X) - y) ** 2.0))

    def R2(self, X=None, y=None):
        if X is None or y is None:
            return self.R2_train
        return 1.0 - self.RMSE(X, y) ** 2.0 / np.average((y - np.average(y)) ** 2.0)


class PCA(object):
    def __init__(self, X):
        self.average = np.average(X, axis=0)
        self.center = X - self.average
        self.cov = self.center.T @ self.center / X.shape[0]
        self.eig_val, self.eig_vec = np.linalg.eigh(self.cov)
        sort_indices = np.argsort(self.eig_val)[::-1]
        self.eig_val = self.eig_val[sort_indices]
        self.eig_vec = self.eig_vec[:, sort_indices]
        self.variance = np.sum(self.eig_val)
        self.percent_var = np.cumsum(self.eig_val) / self.variance

    def scalar_percent(self, percent=0.9):
        return self.center @ self.eig_vec[:, self.percent_var < percent]

    def scalar_num_dim(self, num_dim=2):
        return self.center @ self.eig_vec[:, :num_dim]

    def vector_percent(self, percent=0.9):
        return self.scalar_percent(percent) @ self.eig_vec[:, self.percent_var < percent].T + self.center

    def vector_num_dim(self, num_dim=2):
        return self.scalar_num_dim(num_dim) @ self.eig_vec[:, :num_dim].T + self.center

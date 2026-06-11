import pandas as pd
import numpy as np

import data_loader


class KNN:
    def __init__(self, training_data, k=3, metric='euclidean'):
        assert(metric in ['euclidean', 'cosine', 'manhattan'])
        self.metric = metric
        self.k = k
        self.x = training_data.drop(columns=["subject"]).to_numpy()
        self.y = training_data["subject"].to_numpy()
        self.mus = self.x.mean(axis=0)
        self.sigmas = self.x.std(axis=0)
        self.sigmas = np.where(self.sigmas == 0, 1, self.sigmas)
        self.x = (self.x - self.mus)/self.sigmas

    def predict(self, x):
        x = (x.to_numpy() - self.mus)/self.sigmas
        if self.metric == 'euclidean':
            distances = np.sqrt(np.sum((self.x - x) ** 2, axis=1))
        elif self.metric == 'manhattan':
            distances = np.sum(np.abs(self.x - x), axis=1)
        elif self.metric == 'cosine':
            distances = 1 - np.dot(self.x, x) / (np.linalg.norm(x) * np.linalg.norm(self.x, axis=1))
        else:
            raise Exception("Unknown metric")

        nearest_indices = np.argsort(distances)[:self.k]
        nearest = self.y[nearest_indices]
        print(nearest)


if __name__ == '__main__':
    data = data_loader.load_data(include_rep=False)
    df_train, df_test = data_loader.split_data(data, sample=[(30, 360), (21, 40)], test_bound=40)
    knn = KNN(df_train)
    print(df_test)
    for i in range(10):
        print(df_test['subject'][i])
        knn.predict(df_test.drop(columns=['subject']).iloc[i])
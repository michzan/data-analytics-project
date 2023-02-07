import torch
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


def loadData():
    mean_rating = pd.read_csv('../../csv_genereted/mean_rating.csv')
    movie_genre = pd.read_csv('../../csv_genereted/movies_genre.csv')
    movie_tag_relevance = pd.read_csv('../../csv_genereted/movies_tag_relevance.csv')

    final_df = movie_genre.merge(movie_tag_relevance, on='movieId').merge(mean_rating, on='movieId')
    final_df = final_df.drop('movieId', axis=1)

    return final_df


def preprocess(X):
    standardScaler = StandardScaler()
    X = standardScaler.fit_transform(X)
    pca = PCA(n_components=0.95)
    return pca.fit_transform(X)


class MovieLensDataset(Dataset):
    def __init__(self):
        data = loadData()
        X = data.iloc[:, 1:1149]
        X.columns = X.columns.astype('str')
        y = data.iloc[:, 1149]

        X = preprocess(X)
        
        self.num_input = X.shape[1]
        self.num_output = 1

        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx, :], self.y[idx]




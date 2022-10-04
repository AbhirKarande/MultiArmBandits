#implement linear thompson sampling for contextual linear bandit

import numpy as np
import scipy.stats as stats
import math
import random

class LinTS:
    def __init__(self, dimension, lambda_):
        self.users = {}
        self.dimension = dimension
        self.lambda_ = lambda_

        self.CanEstimateUserPreference = True

    def decide(self, pool_articles, userID):
        if userID not in self.users:
            self.users[userID] = LinTSStruct(self.dimension, self.lambda_)
        return self.users[userID].decide(pool_articles)

    def updateParameters(self, articlePicked, click, userID):
        self.users[userID].updateParameters(articlePicked, click)

    def getTheta(self,userID):
        return self.users[userID].UserTheta


class LinTSStruct:
    def __init__(self, dimension, lambda_):
        self.d = dimension
        self.lambda_ = lambda_

        self.A = np.identity(self.d) * self.lambda_
        self.b = np.zeros((self.d, 1))
        self.UserTheta = np.zeros((self.d, 1))
        self.time = 0

    def updateParameters(self, articlePicked, click):
        x = articlePicked.featureVector
        x = x.reshape(self.d, 1)
        self.A += np.dot(x, x.T)
        self.b += click * x
        self.UserTheta = np.random.multivariate_normal(np.dot(np.linalg.inv(self.A), self.b).reshape(self.d), np.linalg.inv(self.A))
        self.time += 1

    def getTheta(self):
        return self.UserTheta

    def decide(self, pool_articles):
        maxPTA = float('-inf')
        articlePicked = None

        for article in pool_articles:
            x = article.featureVector
            x = x.reshape(self.d, 1)
            pta = np.dot(self.UserTheta.T, x)
            # pick article with highest Prob
            if maxPTA < pta:
                maxPTA = pta
                articlePicked = article
        return articlePicked
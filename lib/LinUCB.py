#implement LinUCB for contextual linear bandit

import numpy as np
import scipy.stats as stats
import math
import random

class LinUCB:
    def __init__(self, dimension, alpha, lambda_):
        self.users = {}
        self.dimension = dimension
        self.alpha = alpha
        self.lambda_ = lambda_

        self.CanEstimateUserPreference = True

    def decide(self, pool_articles, userID):
        if userID not in self.users:
            self.users[userID] = LinUCBStruct(self.dimension, self.alpha, self.lambda_)
        return self.users[userID].decide(pool_articles)

    def updateParameters(self, articlePicked, click, userID):
        self.users[userID].updateParameters(articlePicked, click)

    def getTheta(self,userID):
        return self.users[userID].UserTheta


class LinUCBStruct:
    def __init__(self, dimension, alpha, lambda_):
        self.d = dimension
        self.alpha = alpha
        self.lambda_ = lambda_

        self.A = np.identity(self.d) * self.lambda_
        self.b = np.zeros((self.d, 1))
        self.UserTheta = np.zeros((self.d, 1))
        self.time = 0

    def getTheta(self):
            return self.UserTheta

    def updateParameters(self, articlePicked, click):
        x = articlePicked.featureVector
        x = x.reshape(self.d, 1)
        self.A += np.dot(x, x.T)
        self.b += click * x
        self.UserTheta = np.dot(np.linalg.inv(self.A), self.b)
        self.time += 1
    def decide(self, pool_articles):
        maxPTA = float('-inf')
        articlePicked = None

        for article in pool_articles:
            x = article.featureVector
            x = x.reshape(self.d, 1)
            pta = np.dot(self.UserTheta.T, x) + self.alpha * math.sqrt(np.dot(np.dot(x.T, np.linalg.inv(self.A)), x))
            # pick article with highest Prob
            if maxPTA < pta:
                articlePicked = article
                maxPTA = pta

        return articlePicked
#implement upper confidence bound linear bandit

import numpy as np
import scipy.stats as stats
import math
import random

class UpperConfidenceBoundLinearBandit:
    def __init__(self, dimension, alpha):
        self.users = {}
        self.dimension = dimension
        self.CanEstimateUserPreference = True
        self.alpha = alpha

    def decide(self, pool_articles, userID):
        if userID not in self.users:
            self.users[userID] = UCBLinearStruct(self.dimension, self.alpha)
        return self.users[userID].decide(pool_articles)

    def updateParameters(self, articlePicked, click, userID):
        self.users[userID].updateParameters(articlePicked, click)

    def getTheta(self,userID):
        return self.users[userID].UserArmMean

class UCBLinearStruct:
    def __init__(self, dimension, alpha):
        self.d = dimension
        self.UserArmMean = np.zeros(self.d)
        self.UserArmTrials = np.zeros(self.d)
        self.UserArmVar = np.ones(self.d)
        self.time = 0
        self.alpha = alpha

    def updateParameters(self, articlePicked, click):
        self.UserArmMean += np.dot(np.linalg.inv(self.UserArmVar), articlePicked.features * (click - np.dot(self.UserArmMean, articlePicked.features)))
        self.UserArmVar = np.linalg.inv(np.linalg.inv(self.UserArmVar) + np.dot(articlePicked.features, articlePicked.features.T))
        self.time += 1

    def getTheta(self):
        return self.UserArmMean

    def decide(self, pool_articles):
        maxPTA = float('-inf')
        articlePicked = None

        for article in pool_articles:
            article_pta = np.dot(self.UserArmMean, article.features) + self.alpha * math.sqrt(np.dot(np.dot(article.features, self.UserArmVar), article.features))
            # pick article with highest Prob
            if maxPTA < article_pta:
                articlePicked = article
                maxPTA = article_pta

        return articlePicked
#implement a thompson sampling linear bandit algorithm

import numpy as np
import scipy.stats as stats
import math
import random

class ThompsonSamplingLinearBandit:
    def __init__(self, dimension):
        self.users = {}
        self.dimension = dimension
        self.CanEstimateUserPreference = True

    def decide(self, pool_articles, userID):
        if userID not in self.users:
            self.users[userID] = ThompsonSamplingLinearStruct(self.dimension)
        return self.users[userID].decide(pool_articles)

    def updateParameters(self, articlePicked, click, userID):
        self.users[userID].updateParameters(articlePicked, click)

    def getTheta(self,userID):
        return self.users[userID].UserArmMean



class ThompsonSamplingLinearStruct:
    def __init__(self, dimension):
        self.d = dimension
        self.UserArmMean = np.zeros(self.d)
        self.UserArmTrials = np.zeros(self.d)
        self.UserArmVar = np.ones(self.d)
        self.time = 0
        self.UserArmPrecision = np.ones(self.d)

    def updateParameters(self, articlePicked, click):
        self.UserArmMean += np.dot(articlePicked.id, click - np.dot(self.UserArmMean, articlePicked.id))
        self.UserArmTrials += 1
        self.UserArmVar = 1.0/(self.UserArmTrials+1)
        self.UserArmPrecision = np.dot(self.UserArmTrials, self.UserArmVar)
        self.time += 1

    def getTheta(self):
        return self.UserArmMean

    def decide(self, pool_articles):
        maxPTA = float('-inf')
        articlePicked = None

        for article in pool_articles:
            article_pta = np.dot(self.UserArmMean, article.id) + np.dot(self.UserArmPrecision, np.dot(article.id, np.dot(np.diag(self.UserArmVar), article.id)))
            # pick article with highest Prob
            if maxPTA < article_pta:
                articlePicked = article
                maxPTA = article_pta

        return articlePicked
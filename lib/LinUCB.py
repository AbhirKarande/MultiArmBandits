#implement LinUCB linear bandit algorithm

import numpy as np
import scipy.stats as stats
import math
import random

class LinUCB:
    def __init__(self, dimension, alpha):
        self.users = {}
        self.dimension = dimension
        self.CanEstimateUserPreference = True
        self.alpha = alpha

    def decide(self, pool_articles, userID):
        if userID not in self.users:
            self.users[userID] = LinUCBStruct(self.dimension, self.alpha)
        return self.users[userID].decide(pool_articles)

    def updateParameters(self, articlePicked, click, userID):
        self.users[userID].updateParameters(articlePicked, click)

    def getTheta(self,userID):
        return self.users[userID].UserArmMean


class LinUCBStruct:
    def __init__(self, dimension, alpha):
        self.d = dimension
        self.UserArmMean = np.zeros(self.d)
        self.UserArmTrials = np.zeros(self.d)
        self.UserArmVar = np.ones(self.d)
        self.time = 0
        self.alpha = alpha

    def updateParameters(self, articlePicked, click):
        self.UserArmMean += np.dot(articlePicked.id, click - np.dot(self.UserArmMean, articlePicked.id))
        self.UserArmTrials += 1
        self.UserArmVar = 1.0/(self.UserArmTrials+1)
        self.time += 1

    def getTheta(self):
        return self.UserArmMean

    def decide(self, pool_articles):
        maxPTA = float('-inf')
        articlePicked = None

        for article in pool_articles:
            article_pta = np.dot(self.UserArmMean, article.id) + self.alpha * np.sqrt(np.dot(article.id, np.dot(np.diag(self.UserArmVar), article.id)))
            print(type(article_pta))
            print(article_pta)
            #convert article_pta to float
            article_pta = float(article_pta)
            if maxPTA < article_pta:
                articlePicked = article
                maxPTA = article_pta

        return articlePicked
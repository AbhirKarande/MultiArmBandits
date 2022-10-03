#implement LinTS linear bandit algorithm
#LinTS is a variant of LinUCB, where the variance of the user's preference is estimated by a Bayesian approach
#LinTS is a variant of Thompson Sampling, where the user's preference is estimated by a Bayesian approach

import numpy as np
import scipy.stats as stats
import math
import random

class LinTS:
    def __init__(self, dimension, alpha):
        self.users = {}
        self.dimension = dimension
        self.CanEstimateUserPreference = True
        self.alpha = alpha

    def decide(self, pool_articles, userID):
        if userID not in self.users:
            self.users[userID] = LinTSStruct(self.dimension, self.alpha)
        return self.users[userID].decide(pool_articles)

    def updateParameters(self, articlePicked, click, userID):
        self.users[userID].updateParameters(articlePicked, click)

    def getTheta(self,userID):
        return self.users[userID].UserArmMean


class LinTSStruct:
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
            #sample user's preference from a Gaussian distribution
            theta = np.random.multivariate_normal(self.UserArmMean, np.diag(self.UserArmVar))
            #compute the probability of the article given the user's preference
            article_pta = np.dot(theta, article.id)
            # pick article with highest Prob
            if maxPTA < article_pta:
                articlePicked = article
                maxPTA = article_pta

        return articlePicked
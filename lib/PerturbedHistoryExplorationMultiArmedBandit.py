#implement perturbed history exploration multi-armed bandit

import numpy as np
import scipy.stats as stats
import math
import random

class PerturbedHistoryExplorationMultiArmedBandit:
    def __init__(self, num_arm, alpha):
        self.users = {}
        self.num_arm = num_arm
        self.CanEstimateUserPreference = True
        self.alpha = alpha

    def decide(self, pool_articles, userID):
        if userID not in self.users:
            self.users[userID] = PerturbedHistoryExplorationStruct(self.num_arm, self.alpha)
        return self.users[userID].decide(pool_articles)

    def updateParameters(self, articlePicked, click, userID):
        self.users[userID].updateParameters(articlePicked.id, click)
    
    def getTheta(self,userID):
        return self.users[userID].UserArmMean


class PerturbedHistoryExplorationStruct:
    def __init__(self, num_arm, alpha):
        self.d = num_arm
        self.UserArmMean = np.zeros(self.d)
        self.UserArmTrials = np.zeros(self.d)
        self.UserArmVar = np.ones(self.d)
        self.time = 0
        self.alpha = alpha

    def updateParameters(self, articlePicked_id, click):
        self.UserArmMean[articlePicked_id] = (self.UserArmMean[articlePicked_id]*self.UserArmTrials[articlePicked_id] + click) / (self.UserArmTrials[articlePicked_id]+1)
        self.UserArmTrials[articlePicked_id] += 1
        self.UserArmVar[articlePicked_id] = 1.0/(self.UserArmTrials[articlePicked_id]+1)

        self.time += 1

    def getTheta(self):
        return self.UserArmMean

    def decide(self, pool_articles):
        maxPTA = float('-inf')
        articlePicked = None

        for article in pool_articles:
            article_pta = stats.norm.rvs(loc=self.UserArmMean[article.id], scale=math.sqrt(self.UserArmVar[article.id]))
            # pick article with highest Prob
            if maxPTA < article_pta:
                articlePicked = article
                maxPTA = article_pta

        return articlePicked



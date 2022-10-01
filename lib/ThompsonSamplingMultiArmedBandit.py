#implement thompson sampling multi armed bandit

import numpy as np

class ThompsonSamplingStruct:
    def __init__(self, num_arm):
        self.d=num_arm
        self.UserArmSuccess = np.zeros(self.d)
        self.UserArmFailure = np.zeros(self.d)
        self.time = 0
    def updateParameters(self, articlePicked_id, click):
        if click == 1:
            self.UserArmSuccess[articlePicked_id] += 1
        else:
            self.UserArmFailure[articlePicked_id] += 1
        self.time += 1
    def getTheta(self):
        return self.UserArmSuccess/(self.UserArmSuccess+self.UserArmFailure)
    def decide(self, pool_articles):
        maxPTA = float('-inf')
        articlePicked = None
        for article in pool_articles:
            article_pta = np.random.beta(self.UserArmSuccess[article.id]+1, self.UserArmFailure[article.id]+1)
            # pick article with highest Prob
            if maxPTA < article_pta:
                articlePicked = article
                maxPTA = article_pta
        return articlePicked

class ThompsonSamplingMultiArmedBandit:
    def __init__(self, num_arm):
        self.users = {}
        self.num_arm = num_arm
        self.CanEstimateUserPreference = True
    def decide(self, pool_articles, userID):
        if userID not in self.users:
            self.users[userID] = ThompsonSamplingStruct(self.num_arm)
        return self.users[userID].decide(pool_articles)
    def updateParameters(self, articlePicked, click, userID):
        self.users[userID].updateParameters(articlePicked.id, click)
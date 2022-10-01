#implement perturn history exploration multi armed bandit

import numpy as np

class PerturbedHistoryExplorationMultiArmedBandit:
    def __init__(self, num_arm, perturb_scale):
        self.users = {}
        self.num_arm = num_arm
        self.CanEstimateUserPreference = False
        self.perturb_scale = perturb_scale

    def decide(self, pool_articles, userID):
        if userID not in self.users:
            self.users[userID] = UCBStruct(self.num_arm)
        return self.users[userID].decide(pool_articles)

    def updateParameters(self, articlePicked, click, userID):
        self.users[userID].updateParameters(articlePicked.id, click)
    
    def getTheta(self,userID):
        return self.users[userID].UserArmMean



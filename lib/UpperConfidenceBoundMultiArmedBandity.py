import numpy as np

class UCBStruct:
    def __init__(self, num_arm):
        self.d = num_arm
        self.UserArmMean = np.zeros(self.d)
        self.UserArmTrials = np.zeros(self.d)
        self.time = 0
    def getTheta(self):
        return self.UserArmMean
    
    
    
    def updateParameters(self, articlePicked_id, click):
        # update parameters
        #update the mean of the article picked, 
        # and the number of trials, and the time,  
        self.UserArmMean[articlePicked_id] = (self.UserArmMean[articlePicked_id]*self.UserArmTrials[articlePicked_id] + click) / (self.UserArmTrials[articlePicked_id]+1)
        self.UserArmTrials[articlePicked_id] += 1
        self.time += 1
    
    def decide(self, pool_articles):
        maxPTA = float('-inf')
        articlePicked = None
        #the UCB formula is the mean of the article plus the square root of 2 times the log of the time divided by the number of trials of the article
        for article in pool_articles:
            article_pta = self.UserArmMean[article.id] + np.sqrt(2*np.log(self.time+1)/(self.UserArmTrials[article.id]+1))
            if maxPTA < article_pta:
                articlePicked = article
                maxPTA = article_pta
        return articlePicked



class UCBMultiArmedBandit:
    def __init__(self, num_arm):
        self.users = {}
        self.num_arm = num_arm
        self.CanEstimateUserPreference = False

    def decide(self, pool_articles, userID):
        if userID not in self.users:
            self.users[userID] = UCBStruct(self.num_arm)
        return self.users[userID].decide(pool_articles)

    def updateParameters(self, articlePicked, click, userID):
        self.users[userID].updateParameters(articlePicked.id, click)
    
    def getTheta(self,userID):
        return self.users[userID].UserArmMean
    
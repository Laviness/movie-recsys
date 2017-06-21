import random
import math
import pandas as pd
import numpy as np


class SVD():
    def __init__(self, allfile, trainfile, testfile, factorNum=10):
        # all data file
        self.allfile = allfile
        # training set file
        self.trainfile = trainfile
        # testing set file
        self.testfile = testfile
        # get factor number
        self.factorNum = factorNum
        # get user number
        self.userNum = self.getUserNum()
        # get item number
        self.itemNum = self.getItemNum()
        # learning rate
        self.learningRate = 0.01
        # the regularization lambda
        self.regularization = 0.05
        # initialize the model and parameters
        self.initModel()
        # get user number function

    def getUserNum(self):
        file = self.allfile
        return file.user_id.unique().shape[0]
        # get item number function

    def getItemNum(self):
        file = self.allfile
        return file.item_id.max()
        # initialize all parameters

    def initModel(self):
        self.av = self.average(self.trainfile)
        self.bu = [0.0 for i in range(self.userNum)]
        self.bi = [0.0 for i in range(self.itemNum)]
        temp = math.sqrt(self.factorNum)
        self.pu = [[(0.1 * random.random() / temp) for i in range(self.factorNum)] for j in range(self.userNum)]
        self.qi = [[0.1 * random.random() / temp for i in range(self.factorNum)] for j in range(self.itemNum)]
        print("Initialize end.The user number is:%d,item number is:%d,the average score is:%f" % (self.userNum, self.itemNum, self.av))
        # train model

    def train(self, iterTimes=100):
        print("Beginning to train the model......")
        trainfile = self.trainfile
        preRmse = 10000.0
        for iter in range(iterTimes):
            # read the training file
            for line in trainfile.itertuples():
                #content = line.split('\n')
                user = int(line[1]) - 1
                item = int(line[2]) - 1
                rating = float(line[3])
                # calculate the predict score
                pscore = self.predictScore(self.av, self.bu[user], self.bi[item], self.pu[user], self.qi[item])
                # the delta between the real score and the predict score
                eui = rating - pscore

                # update parameters bu and bi(user rating bais and item rating bais)
                self.bu[user] += self.learningRate * (eui - self.regularization * self.bu[user])
                self.bi[item] += self.learningRate * (eui - self.regularization * self.bi[item])
                for k in range(self.factorNum):
                    temp = self.pu[user][k]
                    # update pu,qi
                    self.pu[user][k] += self.learningRate * (
                    eui * self.qi[user][k] - self.regularization * self.pu[user][k])
                    self.qi[item][k] += self.learningRate * (temp * eui - self.regularization * self.qi[item][k])
                    # print pscore,eui
            # close the file
            #fi.close()
            # calculate the current rmse
            curRmse = self.test(self.av, self.bu, self.bi, self.pu, self.qi)
            print ("Iteration %d times,RMSE is : %f" % (iter + 1, curRmse))
            if curRmse > preRmse:
                break
            else:
                preRmse = curRmse
        print("Iteration finished!")
        # test on the test set and calculate the RMSE

    def test(self, av, bu, bi, pu, qi):
        testfile = self.testfile
        rmse = 0.0
        cnt = 0
        for line in testfile.itertuples():
            cnt += 1
            user = int(line[1]) - 1
            item = int(line[2]) - 1
            score = float(line[3])
            pscore = self.predictScore(av, bu[user], bi[item], pu[user], qi[item])
            rmse += math.pow(score - pscore, 2)
        return math.sqrt(rmse / cnt)
        # calculate the average rating in the training set

    def average(self, filename):
        # result = 0.0
        # cnt = 0
        # filename.rating.sum()
        #
        # for line in filename:
        #     cnt += 1
        #     score = float(line.split('\ ')[2])
        #     result += score
        return filename.rating.mean()
        # calculate the inner product of two vectors

    def innerProduct(self, v1, v2):
        result = 0.0
        for i in range(len(v1)):
            result += v1[i] * v2[i]
        return result

    def predictScore(self, av, bu, bi, pu, qi):
        pscore = av + bu + bi + self.innerProduct(pu, qi)
        if pscore < 1:
            pscore = 1
        if pscore > 5:
            pscore = 5
        return pscore

    def generate(self):
        file_generate = 'full_data_file.txt'
        full_data_matrix = np.zeros((self.userNum, self.itemNum))
        for user in range(self.userNum):
            for item in range(self.itemNum):
                pscore=float(self.predictScore(self.av, self.bu[user], self.bi[item], self.pu[user], self.qi[item]))
                full_data_matrix[user, item]=pscore
                #print(str(user+1)+' '+str(item+1)+' '+str(pscore)+'\n')
                with open(file_generate, "a") as f:
                    f.write(str(user+1)+' '+str(item+1)+' '+str(pscore)+"\n")

        #np.savetxt('full_data.txt', full_data_matrix)
        return 0


if __name__ == '__main__':
    header = ['user_id', 'item_id', 'rating']

    movie_data = pd.read_csv('train_all_txt/train_all_txt.txt', sep='\ ', names= header)

    from sklearn import cross_validation as cv
    train_data, test_data = cv.train_test_split(movie_data, test_size=0.2)
    np.savetxt('train_data.txt', train_data)
    np.savetxt('test_data.txt', test_data)
    s = SVD(movie_data, train_data, test_data)
    #s = SVD(movie_data, "train.txt", "test.txt")
    # print s.userNum,s.itemNum
    # print s.average("data\\ua.base")
    s.train()
    s.generate()
    # print(s.bu)
    # print(s.bi)
    # print(s.pu)
    # print(s.qi)
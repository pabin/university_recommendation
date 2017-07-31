import pandas as pd
import numpy as np
import csv
import random
import math
import operator


def k_nearest_neighbor_from_scratch():
    # user = User.objects.get(username=request.user.username)
    # all_student_info = student_info.objects.filter(user=user)

    # for major in all_student_info:
    #     intended_major = major.Intended_Major
    #     student_gpa = major.UnderGrad_GPA
    #     student_verbal_score = major.GRE_Verbal_Score
    #     student_quant_score = major.GRE_Quant_Score
    #     student_awa_score = major.GRE_AWA_Score
    #     toefl_score = major.TOEFL_Score
    #     status = major.Student_Status
    #     degree = major.Degree_applying

    csv_filepathname = "/home/raj/PycharmProjects/university_recommendation/" \
                       "database/clean238k.csv"

    names = ['University', 'Major', 'Degree', 'Season', 'Decision', 'GPA',
             'Verbal', 'Quant', 'AWA', 'TOEFL', 'Status', 'Comments', 'Comments_Cont']

    df = pd.read_csv(csv_filepathname, names=names, header=None)

    df1 = df[(df['Major'] == 'Computer Science') &
             (df['Degree'] == 'MS') & (df['Status'] == 'International')]

    df2 = df1.drop(['Major', 'Degree', 'Season', 'University', 'Status', 'Comments', 'Comments_Cont',], 1)
    df2 = df2[['GPA', 'Verbal', 'Quant', 'AWA', 'TOEFL', 'Decision']]
    #df2 = df2.head(1500)
    print df2
    def handle_non_numeric_data(df):
        columns = df.columns.values

        for column in columns:
            text_digit_vals = {}
            def convert_to_int(val):
                return text_digit_vals[val]


            if df[column].dtype != np.int64 and df[column].dtype !=np.float64:
                column_contents =df[column].values.tolist()
                unique_elements = set(column_contents)
                x = 0
                for unique in unique_elements:
                    if unique not in text_digit_vals:
                        text_digit_vals[unique] = x
                        x+=1

                df[column] = list(map(convert_to_int, df[column]))

        return df 

    df2 = handle_non_numeric_data(df2)
    df2.fillna(0, inplace=True)
    df2 = df2.dropna()

    print df2

    final_data = df2.values.tolist()

    def loadDataset(filename, split, trainingSet=[] , testSet=[]):
        #with open(filename, 'rb') as csvfile:
            #lines = csv.reader(csvfile)
            dataset = final_data
            for x in range(len(dataset)-1):
                for y in range(4):
                    dataset[x][y] = float(dataset[x][y])
                if random.random() < split:
                    trainingSet.append(dataset[x])
                else:
                    testSet.append(dataset[x])


    def euclideanDistance(instance1, instance2, length):
        distance = 0
        for x in range(length):
            distance += pow((instance1[x] - instance2[x]), 2)
        return math.sqrt(distance)

    def getNeighbors(trainingSet, testInstance, k):
        distances = []
        length = len(testInstance)-1
        for x in range(len(trainingSet)):
            dist = euclideanDistance(testInstance, trainingSet[x], length)
            distances.append((trainingSet[x], dist))
        distances.sort(key=operator.itemgetter(1))
        neighbors = []
        for x in range(k):
            neighbors.append(distances[x][0])
        return neighbors

    def getResponse(neighbors):
        classVotes = {}
        for x in range(len(neighbors)):
            response = neighbors[x][-1]
            if response in classVotes:
                classVotes[response] += 1
            else:
                classVotes[response] = 1
        sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
        return sortedVotes[0][0]

    def getAccuracy(testSet, predictions):
        correct = 0
        for x in range(len(testSet)):
            if testSet[x][-1] == predictions[x]:
                correct += 1
        return (correct/float(len(testSet))) * 100.0
        
    def main():
        # prepare data
        trainingSet=[]
        testSet=[]
        split = 0.80
        loadDataset('iris.data', split, trainingSet, testSet)
        print 'Train set: ' + repr(len(trainingSet))
        print 'Test set: ' + repr(len(testSet))
        # generate predictions
        predictions=[]
        k = 3
        for x in range(len(testSet)):
            neighbors = getNeighbors(trainingSet, testSet[x], k)
            result = getResponse(neighbors)
            predictions.append(result)
            print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
        accuracy = getAccuracy(testSet, predictions)
        print('Accuracy: ' + repr(accuracy) + '%')
        
    main()

k_nearest_neighbor_from_scratch()
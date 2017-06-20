from accounts.models import student_info
from django.shortcuts import render
from accounts.models import User
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split


# --k-nearest Neighbour Classifier--
def knn_model(request):
    user = User.objects.get(username=request.user.username)
    all_student_info = student_info.objects.filter(user=user)

    for major in all_student_info:
        intended_major = major.Intended_Major
        student_gpa = major.UnderGrad_GPA
        student_verbal_score = major.GRE_Verbal_Score
        student_quant_score = major.GRE_Quant_Score
        student_awa_score = major.GRE_AWA_Score
        status = major.Student_Status

    print status, intended_major, student_gpa, student_verbal_score, \
          student_quant_score, student_awa_score

    csv_filepathname = "/home/raj/PycharmProjects/university_recommendation/" \
                       "database/Final_College_Datasets.csv"

    names = ['University_Name', 'Major', 'Degree', 'Season', 'class', 'Undergrad_GPA',
             'GRE_Verbal', 'GRE_Quant', 'GRE_AWA', 'Status']

    # names = ['University', 'Major', 'Degree', 'Season', 'class', 'GPA',
    #        'Verbal', 'Quant', 'AWA', 'TOEFL', 'Status', 'Comments', 'Comments_Cont']

    df = pd.read_csv(csv_filepathname, names=names, header=None)

    # df1 = df[(df['Major'] == intended_major) & (df['Decision'] == 'Accepted')]

    df1 = df[(df['Major'] == intended_major) &
             (df['Degree'] == 'MS') & (df['Status'] == status)]

    # create design matrix X and target vector y
    X = np.array(df1.ix[:, 5:9])  # end index is exclusive
    y = np.array(df1['class'])  # another way of indexing a pandas df
    z = np.array(df1.ix[:, 0:5])
    z1 = z[:, np.array([True, True, False, False, True])]
    # print X

    # --split into train and test--
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.20, random_state=42)

    # instantiate learning model (k = 4)
    knn = KNeighborsClassifier(n_neighbors=6)

    # fitting the model
    knn.fit(X_train, y_train)

    # predict the response
    pred = knn.predict(X_test)

    train = np.array(zip(X_train, y_train))
    test = np.array(zip(X_test, y_test))

    # evaluate accuracy
    print "Accuracy Percent is:", accuracy_score(y_test, pred)
    print "------------"

    example = np.array([student_gpa, student_verbal_score,
                        student_quant_score, student_awa_score])
    example = example.reshape(1, -1)

    prediction = knn.predict(example)
    print "Best_Recommendation: ", (prediction)
    ind = knn.kneighbors(example)
    print ind
    # print ind[1]
    print "------------"

    # print X[ind[1]]
    z2 = z1[ind[1]]
    # print z2

    print "-----Recommended Universities:-----"

    z4 = np.array([])
    for counter in range(0, 6, 1):
        if 'Accepted' in z2[0, counter]:
            z3 = z2[0, counter]
            z4 = np.append(z4, z3)

            # print z4
    recommend = np.array([])
    for count in range(0, len(z4), 3):
        recommended = z4[count]
        recommend = np.append(recommend, recommended)
    # print recommend

    list = []
    for i in recommend:
        if i not in list:
            list.append(i)
    print list

    context = {'list': list}
    return render(request, 'knn.html', context)

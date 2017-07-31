from accounts.models import student_info
from django.shortcuts import render
from accounts.models import User
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from scipy.interpolate import spline
from sklearn.cross_validation import train_test_split, cross_val_score
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')


def knn_model(request):
    user = User.objects.get(username=request.user.username)
    all_student_info = student_info.objects.filter(user=user)

    for major in all_student_info:
        intended_major = major.Intended_Major
        student_gpa = float(major.UnderGrad_GPA)
        student_gpa_normalized = (student_gpa - 0.25) / (4 - 0.25)

        student_verbal_score = float(major.GRE_Verbal_Score)
        student_verbal_score_normalized = (student_verbal_score - 130.0) / (170.0 - 130.0)

        student_quant_score = float(major.GRE_Quant_Score)
        student_quant_score_normalized = (student_quant_score - 131.0) / (170.0 - 131.0)

        student_awa_score = float(major.GRE_AWA_Score)
        student_awa_score_normalized = (student_awa_score - 0.30) / (6.0 - 0.30)

        toefl_score = float(major.TOEFL_Score)
        toefl_score_normalized = (toefl_score - 57.0) / (120.0 - 57.0)

        example = np.array([student_gpa_normalized, student_verbal_score_normalized,
                            student_quant_score_normalized, student_awa_score_normalized, toefl_score_normalized])
        example = example.reshape(1, -1)

    csv_filepathname = "/home/raj/PycharmProjects/university_recommendation/" \
                       "database/normal130k.csv"

    best_university_prediction = best_university_prediction_using_knn(csv_filepathname, intended_major,
                                                                      example)

    admission_prediction_details = admission_prediction_using_knn(csv_filepathname, intended_major, example)
    university_where_accepted = admission_prediction_details[0]
    university_where_rejected = admission_prediction_details[1]
    admission_prediction = admission_prediction_details[2]

    # universities_for_recommendation =
    # k_means_clustering_for_similar_universities(best_university_prediction, intended_major)

    context = {
        # 'universities_for_recommendation': universities_for_recommendation,
        'best_university_prediction': best_university_prediction,
        'university_where_accepted': university_where_accepted,
        'university_where_rejected': university_where_rejected,
        'admission_prediction': admission_prediction
    }

    return render(request, 'knn.html', context)


# ---------------------------------------------------------------------------------
# -----------------------KNN For Best University Prediction------------------------

def best_university_prediction_using_knn(csv_filepathname, intended_major, example):
    names = ['class', 'Major', 'Degree', 'Season', 'Decision', 'GPA',
             'Verbal', 'Quant', 'AWA', 'TOEFL']

    df = pd.read_csv(csv_filepathname, names=names, header=None)

    df.fillna(0, inplace=True)
    # df1 = df[(df['Decision'] == 'Accepted')]
    df1 = df[(df['Major'] == intended_major) & (df['Decision'] == 'Accepted')]
    df4 = df1
    df5 = df1
    df6 = df1
    df7 = df1
    df8 = df1
    df10 = df1
    df11 = df1
    df12 = df1
    df13 = df1
    df14 = df1
    df15 = df1

    frames = [df1, df4, df5, df6, df7, df8, df10, df11, df12, df13, df14, df15]
    df1 = pd.concat(frames)

    # ---create design matrix X and target vector y---
    X = np.array(df1.ix[:, 5:10])  # end index is exclusive
    X = preprocessing.scale(X)

    y = np.array(df1['class'])  # another way of indexing a pandas df

    # --split into train and test--
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=32)

    # --instantiate learning model (k = 4)--
    knn_for_university = KNeighborsClassifier(n_neighbors=15)

    uni_scores = cross_val_score(knn_for_university, X, y, cv=5, scoring='accuracy')
    cross_validated_accuracy_university_prediction = uni_scores.mean()

    print ('--Cross Validated Accuracy', cross_validated_accuracy_university_prediction)

    # --fitting the model--
    knn_for_university.fit(X_train, y_train)

    # --predict the response--
    predicted = knn_for_university.predict(X_test)

    # --evaluate accuracy--
    print ('Accuracy % for University Recommendation:', accuracy_score(y_test, predicted))
    # example = np.array([student_gpa_normalized, student_verbal_score_normalized,
    #                     student_quant_score_normalized, student_awa_score_normalized, toefl_score_normalized])
    # example = example.reshape(1, -1)

    best_university_prediction = knn_for_university.predict(example)
    print ("Best Recommended University: ", best_university_prediction)

    return (best_university_prediction)


# ---------------------------------------------------------------------------------
# --------------------------KNN For Admission Prediction---------------------------

def admission_prediction_using_knn(csv_filepathname, intended_major, example):
    names = ['University', 'Major', 'Degree', 'Season', 'class', 'GPA',
             'Verbal', 'Quant', 'AWA', 'TOEFL']

    db = pd.read_csv(csv_filepathname, names=names, header=None)
    db.fillna(0, inplace=True)
    db1 = db[(db['Major'] == intended_major)]
    db09 = db1
    db10 = db1
    db11 = db1
    db12 = db1
    db13 = db1
    db14 = db1
    db15 = db1
    frames1 = [db1, db09, db10, db11, db12, db13, db14, db15]
    db1 = pd.concat(frames1)

    # ---create design matrix X and target vector y---
    x1 = np.array(db1.ix[:, 5:10])  # end index is exclusive
    x1 = preprocessing.scale(x1)
    y1 = np.array(db1['class'])  # another way of indexing a pandas df
    z2 = np.array(db1.ix[:, 0:5])
    z11 = z2[:, np.array([True, True, False, False, True])]

    # --split into train and test--
    x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.20, random_state=4)

    def check_value_of_k_for_knn(x_data, y_data):
        k_range = range(1, 31, 1)
        k_scores = []
        for k in k_range:
            knn = KNeighborsClassifier(k, weights='uniform')
            scores = cross_val_score(knn, x_data, y_data, cv=25, scoring='accuracy')
            k_scores.append(scores.mean())
        # print ('-----Scores:', k_scores)

        xnew = np.array(k_range)
        ynew = np.array(k_scores)
        xnew_smooth = np.linspace(xnew.min(), xnew.max(), 400)
        ynew_smooth = spline(xnew, ynew, xnew_smooth)
        plt.plot(xnew_smooth, ynew_smooth)

        # plt.plot(k_range, k_scores)
        plt.xlabel('Value of K for KNN')
        plt.ylabel('Cross-Validation Accuracy')
        plt.title('Variation of accuracy with nearest neighbours')
        plt.show()
        return

    #check_value_of_k_for_knn(x1, y1)

    # --instantiate learning model (k = 15)--
    knn_for_admission = KNeighborsClassifier(n_neighbors=15)
    scores = cross_val_score(knn_for_admission, x1, y1, cv=25, scoring='accuracy')
    cross_validated_accuracy = scores.mean()
    print ('--Cross Validated Accuracy', cross_validated_accuracy)

    # --fitting the model--
    knn_for_admission.fit(x1_train, y1_train)

    # --predict the response--
    admission_predicted = knn_for_admission.predict(x1_test)

    # --evaluate accuracy--
    accuracy_for_admission_predicted = accuracy_score(y1_test, admission_predicted)
    print ('Accuracy % for Admisssion Prediction:', accuracy_for_admission_predicted)

    admission_prediction = knn_for_admission.predict(example)
    print ("Your Admission Prediction: ", admission_prediction)

    distance_and_index_of_nearest_neighbors = knn_for_admission.kneighbors(example)
    print (distance_and_index_of_nearest_neighbors)
    uni_name_for_acp_and_rej = z11[distance_and_index_of_nearest_neighbors[1]]
    print ("-----------")
    only_accepted = np.array([])
    only_rejected = np.array([])

    # ---Distinguish Accepted and Rejected Decision for university---
    for counter in range(0, 12, 1):
        if 'Accepted' in uni_name_for_acp_and_rej[0, counter]:
            count_accepted = uni_name_for_acp_and_rej[0, counter]
            only_accepted = np.append(only_accepted, count_accepted)
        if 'Rejected' in uni_name_for_acp_and_rej[0, counter]:
            count_rejected = uni_name_for_acp_and_rej[0, counter]
            only_rejected = np.append(only_rejected, count_rejected)

    # ---Function to Isolate and remove duplicate University---
    def remove_duplicates(university_list):
        isolated_list = np.array([])
        for count in range(0, len(university_list), 3):
            isolation_count = university_list[count]
            isolated_list = np.append(isolated_list, isolation_count)

        # ---Remove Duplicate Universities----
        final_list = []
        for i in isolated_list:
            if i not in final_list:
                final_list.append(i)

        return final_list

    university_where_accepted = remove_duplicates(only_accepted)
    university_where_rejected = remove_duplicates(only_rejected)

    return university_where_accepted, university_where_rejected, admission_prediction


# ---------------------------------------------------------------------------------
# --------------------------K Means Clustering for Finding Similar Universities---------------------------

def k_means_clustering_for_similar_universities(best_university_predicted, intended_major):

    csv_filepathname = "/home/raj/PycharmProjects/university_recommendation/" \
                       "database/clean130k.csv"
    names = ['University', 'Major', 'Degree', 'Season', 'Decision', 'GPA',
             'Verbal', 'Quant', 'AWA', 'TOEFL']
    for university in best_university_predicted:
        best_university = university

    df = pd.read_csv(csv_filepathname, names=names, header=None)

    def get_university_details_for_clustering(university, intended_major, df):
        df1 = df[(df['University'] == university) & (df['Major'] == intended_major)]
        df2 = df1[(df1['Decision'] == 'Accepted')]
        df2['GRE_Scores'] = (df2[['Verbal', 'Quant']].sum(axis=1))

        # df2['GRE_Score'] == df2['Verbal'] + df2['Quant']
        average_gre = (df2['GRE_Scores'].dropna()).mean()
        min_gre = (df2['GRE_Scores'].dropna()).min()
        max_gre = (df2['GRE_Scores'].dropna()).max()

        average_gre_verbal = (df2['Verbal'].dropna()).mean()
        min_gre_verbal = (df2['Verbal'].dropna()).min()
        max_gre_verbal = (df2['Verbal'].dropna()).max()

        average_gre_quant = (df2['Quant'].dropna()).mean()
        min_gre_quant = (df2['Quant'].dropna()).min()
        max_gre_quant = (df2['Quant'].dropna()).max()

        average_gpa = (df2['GPA'].dropna()).mean()
        min_gpa = (df2['GPA'].dropna()).min()
        max_gpa = (df2['GPA'].dropna()).max()

        average_writing = (df2['AWA'].dropna()).mean()
        min_writing = (df2['AWA'].dropna()).min()
        max_writing = (df2['AWA'].dropna()).max()

        average_toefl = (df2['TOEFL'].dropna()).mean()
        min_toefl = (df2['TOEFL'].dropna()).min()
        max_toefl = (df2['TOEFL'].dropna()).max()

        total_accepted = df1[df1['Decision'] == 'Accepted'].Decision.count()
        total_rejected = df1[df1['Decision'] == 'Rejected'].Decision.count()

        acceptance_rate = float((total_accepted / float(total_accepted + total_rejected)) * 100)

        info_list = [acceptance_rate, average_gre, min_gre, max_gre, average_gre_verbal, min_gre_verbal,
                     max_gre_verbal, average_gre_quant, min_gre_quant, max_gre_quant, average_gpa, min_gpa,
                     max_gpa, average_writing, min_writing, max_writing, average_toefl, min_toefl, max_toefl]

        return info_list

    example = get_university_details_for_clustering(best_university, intended_major, df)
    print ('this on', example)

    universities = df.University

    total_universities_list = []
    for university in universities:
        if university not in total_universities_list:
            total_universities_list.append(university)

    record = {}
    for query in total_universities_list:
        record[query] = get_university_details_for_clustering(query, intended_major, df)

    names = ['acceptance_rate', 'average_gre', 'min_gre', 'max_gre', 'average_gre_verbal',
             'min_gre_verbal', 'max_gre_verbal', 'average_gre_quant', 'min_gre_quant', 'max_gre_quant',
             'average_gpa', 'min_gpa', 'max_gpa', 'average_writing', 'min_writing', 'max_writing',
             'average_TOEFL', 'min_TOEFL', 'max_TOEFL']

    new_data = pd.DataFrame.from_dict(record, orient='index')
    final_data = new_data.dropna()
    final_data.columns = names
    final_data['University'] = final_data.index

    X = np.array(final_data.ix[:, 0:19])
    clf = KMeans(n_clusters=40)
    clf.fit(X)

    cluster_map = pd.DataFrame()
    cluster_map['University'] = final_data.University.values
    cluster_map['cluster'] = clf.labels_

    cluster_number = clf.predict(example)

    universities_and_cluster_number = cluster_map[np.array(cluster_map.cluster) == cluster_number]
    required_universities_for_recommendation = np.array(universities_and_cluster_number['University'])

    print (required_universities_for_recommendation)

    return required_universities_for_recommendation

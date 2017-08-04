from django.shortcuts import render
from accounts.models import User
from accounts.models import student_info
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from fuzzywuzzy import process
from bokeh.resources import CDN
from bokeh.embed import components
from bokeh.models import Legend
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from bokeh.plotting import *
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import svm


def home(request):
    query = request.GET.get('q')
    # print query

    context = search_university(query)

    return render(request, 'home.html', context)


# ---------------------------------------------------------------------------------
# --------------------------University Search function-----------------------------

def search_university(query):

    csv_filepathname = "/home/raj/PycharmProjects/university_recommendation/" \
                       "database/clean130k.csv"

    names = ['University', 'Major', 'Degree', 'Season', 'Decision', 'GPA',
             'Verbal', 'Quant', 'AWA', 'TOEFL']

    df = pd.read_csv(csv_filepathname, names=names, header=None)

    if query is not None:

        universities = df.University

        total_universities_list = []
        for university in universities:
            if university not in total_universities_list:
                total_universities_list.append(university)

        # university_name.to_csv('all_universities_list.csv')
        final_university = process.extractOne(query, total_universities_list)
        if final_university[1] < 80:
            query = None
        else:
            query = final_university[0]

        print ('ok here', query)

    df1 = df[df['University'] == query]
    df2 = df1[(df1['Decision'] == 'Accepted')]

    # print df2
    if df2 is not None:

        df2['GRE_Scores'] = (df2[['Verbal', 'Quant']].sum(axis=1))

        avg_gre = (df2['GRE_Scores'].dropna()).mean()
        average_gre = float("%.2f" % avg_gre)
        min_gre = (df2['GRE_Scores'].dropna()).min()
        max_gre = (df2['GRE_Scores'].dropna()).max()

        average_gre_verbal = (df2['Verbal'].dropna()).mean()
        min_gre_verbal = (df2['Verbal'].dropna()).min()
        max_gre_verbal = (df2['Verbal'].dropna()).max()

        average_gre_quant = (df2['Quant'].dropna()).mean()
        min_gre_quant = (df2['Quant'].dropna()).min()
        max_gre_quant = (df2['Quant'].dropna()).max()

        avg_gpa = (df2['GPA'].dropna()).mean()
        average_gpa = float("%.2f" % avg_gpa)
        min_gpa = (df2['GPA'].dropna()).min()
        max_gpa = (df2['GPA'].dropna()).max()

        avg_toefl = (df2['TOEFL'].dropna()).mean()
        average_toefl = float("%.2f" % avg_toefl)
        min_toefl = (df2['TOEFL'].dropna()).min()
        max_toefl = (df2['TOEFL'].dropna()).max()

        total_accepted = df1[df1['Decision'] == 'Accepted'].Decision.count()
        total_rejected = df1[df1['Decision'] == 'Rejected'].Decision.count()
        total_wait_listed = df1[df1['Decision'] == 'Wait listed'].Decision.count()
        total_interview = df1[df1['Decision'] == 'Interview'].Decision.count()
        total_other_decision = df1[df1['Decision'] == 'Other'].Decision.count()

        accept_rate = float((total_accepted/float(total_accepted + total_rejected +
                                total_wait_listed + total_interview + total_other_decision)))
        acceptance_rate = float("%.2f" % (accept_rate*100))

        majors = df1.Major

        final_majors = []
        for i in majors:
            if i not in final_majors:
                final_majors.append(i)

        total_majors = len(final_majors)

        def create_graph_for_scores(average, max, min, min_range, max_range):

            plot = figure(plot_width=600, plot_height=350, y_range=(min_range, max_range))

            bar1 = plot.vbar(x=[1], width=0.5, bottom=0, top=[average], color=["blue"], alpha=0.7)

            bar2 = plot.vbar(x=[2], width=0.5, bottom=0, top=[max], color=["green"], alpha=0.7)

            bar3 = plot.vbar(x=[3], width=0.5, bottom=0, top=[min], color=["red"], alpha=0.7)

            plot.yaxis.axis_label = "GRE Scores"

            legend = Legend(items=[
                ("AVERAGE", [bar1]),
                ("MAXIMUM", [bar2]),
                ("MINIMUM", [bar3]),
            ], location=(0, -30))

            plot.add_layout(legend, 'right')

            script, div = components(plot, CDN)

            return script, div

        graph_for_gre = create_graph_for_scores(average_gre, max_gre, min_gre, min_range=260, max_range=340)
        graph_for_gpa = create_graph_for_scores(average_gpa, max_gpa, min_gpa, min_range=0, max_range=4)
        graph_for_toefl = create_graph_for_scores(average_toefl, max_toefl, min_toefl, min_range=20, max_range=120)

        # data = pd.Series([acceptance_rate, float(1-acceptance_rate)], index=['Acceptance Rate', 'Rejection Rate'])
        # plot = Donut(data)
        #
        # script, div = components(plot, CDN)

        # # define starts/ends for wedges from percentages of a circle
        # percents = [0, acceptance_rate, 100-acceptance_rate]
        # starts = [p * 2 * pi for p in percents[:-1]]
        # ends = [p * 2 * pi for p in percents[1:]]
        #
        # # a color for each pie piece
        # colors = ["yellow", "red"]
        #
        # plot = figure(x_range=(-1, 1), y_range=(-1, 1), plot_width=430, plot_height=360)
        #
        # plot.wedge(x=0, y=0, radius=0.80, start_angle=starts, end_angle=ends, color=colors, alpha=0.7)
        #
        # script, div = components(plot, CDN)


        context = {
            'script_for_gpa': graph_for_gpa[0],
            'div_for_gpa': graph_for_gpa[1],
            'script_for_toefl': graph_for_toefl[0],
            'div_for_toefl': graph_for_toefl[1],
            'script_for_gre': graph_for_gre[0],
            'div_for_gre': graph_for_gre[1],
            # 'script': script,
            # 'div': div,
            'university_name': query,
            'acceptance_rate': acceptance_rate,
            'average_gre': average_gre,
            'min_gre': min_gre,
            'max_gre': max_gre,
            'average_gre_verbal': average_gre_verbal,
            'average_gre_quant': average_gre_quant,
            'average_gpa': average_gpa,
            'total_majors': total_majors,
            'final_majors': final_majors,
            'min_gre_verbal': min_gre_verbal,
            'max_gre_verbal': max_gre_verbal,
            'min_gre_quant': min_gre_quant,
            'max_gre_quant': max_gre_quant,
            'min_gpa': min_gpa,
            'max_gpa': max_gpa,
            'average_toefl': average_toefl,
            'min_toefl': min_toefl,
            'max_toefl': max_toefl
        }
    else:
        context = {
            'error_message': 'Sorry! University Not found.'
        }
    return context


# ---------------------------------------------------------------------------------
# --------------------------KNN For Admission Prediction---------------------------

def admission_prediction_using_knn(request):
    query = request.GET.get('q')

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

    csv_filepathname = "/home/raj/PycharmProjects/university_recommendation/" \
                       "database/clean130k.csv"

    names = ['University', 'Major', 'Degree', 'Season', 'class', 'GPA',
             'Verbal', 'Quant', 'AWA', 'TOEFL']

    db = pd.read_csv(csv_filepathname, names=names, header=None)
    db.fillna(0, inplace=True)

    # majors = db.Major
    #
    # total_major_list = []
    # for major in majors:
    #     if major not in total_major_list:
    #         total_major_list.append(major)
    #
    # # university_name.to_csv('all_universities_list.csv')
    # final_major = process.extractOne(query, total_major_list)
    # if final_major[1] < 80:
    #     query = None
    # else:
    #     intended_major = final_major[0]
    # print ('here', intended_major)

    db1 = db[(db['Major'] == intended_major) & (db['University'] == query)]
    db11 = db1
    db12 = db1
    db13 = db1
    db14 = db1
    db15 = db1

    frames1 = [db1, db11, db12, db13, db14, db15]
    db1 = pd.concat(frames1)

    if len(db1) > 25:
        # ---create design matrix X and target vector y---
        x1 = np.array(db1.ix[:, 5:10])  # end index is exclusive
        y1 = np.array(db1['class'])  # another way of indexing a pandas df
        z2 = np.array(db1.ix[:, 0:5])
        z11 = z2[:, np.array([True, True, False, False, True])]

        # --split into train and test--
        x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.20, random_state=4)

        # --instantiate learning model (k = 15)--
        knn_for_admission = KNeighborsClassifier(n_neighbors=15)
        scores = cross_val_score(knn_for_admission, x1, y1, cv=25, scoring='accuracy')
        cross_validated_accuracy = scores.mean()
        print ('--------------------------------------------------')

        # print ('--Cross Validated Accuracy_KNN', cross_validated_accuracy)
        print ('--------------------------------------------------')

        # --fitting the model--
        knn_for_admission.fit(x1_train, y1_train)

        example = np.array([student_gpa_normalized, student_verbal_score_normalized,
                            student_quant_score_normalized, student_awa_score_normalized, toefl_score_normalized])
        example = example.reshape(1, -1)

        # ----Predicting the admission by model----
        admission_prediction = knn_for_admission.predict(example)
        print ('--------------------------------------------------')

        print ("Admission_Prediction_KNN: ", admission_prediction)
        print ('--------------------------------------------------')

        admission_decision_accuracy = float("%.2f" % (cross_validated_accuracy*100))

        if admission_prediction == "Accepted":
            accepted_percent = admission_decision_accuracy
            rejected_percent = 100 - admission_decision_accuracy
        else:
            rejected_percent = admission_decision_accuracy

            accepted_percent = 100 - admission_decision_accuracy

        plot = figure(plot_width=600, plot_height=450, y_range=(0, 100))

        # plot.circle_cross([1, 2, 3, 4], [2, 4, 1, 2], size=20, color="green", alpha=1)
        # plot.line([1, 2, 3, 4], [2, 4, 1, 2], line_width=2, color="black", alpha=0.5)

        # plot.vbar(x=[1], width=0.5, bottom=0, top=[100], color=["blue"], alpha=0.7)

        bar1 = plot.vbar(x=[2], width=0.5, bottom=0, top=[accepted_percent],
                  color=["green"], alpha=0.7)

        bar2 = plot.vbar(x=[3], width=0.5, bottom=0, top=[rejected_percent],
                  color=["red"], alpha=0.7)

        plot.yaxis.axis_label = "Percent"

        legend = Legend(items=[
            ("Acceptance Chance", [bar1]),
            ("Rejection Chance", [bar2]),
        ], location=(0, -30))

        plot.add_layout(legend, 'right')

        # area.y_range = Range1d(0, 270)

        script, div = components(plot, CDN)

        # random_forest_decision = random_forest_for_admission_prediction(csv_filepathname, query,
        #                                                                 intended_major, example)
        #
        # rf_decision_script = random_forest_decision[0]
        # rf_decision_div = random_forest_decision[1]

        svm_admission_decision = support_vector_machine(csv_filepathname, query, intended_major, example)

        svm_decision_script = svm_admission_decision[0]
        svm_decision_div = svm_admission_decision[1]

        context = {
            # 'rf_decision_script': rf_decision_script,
            # 'rf_decision_div': rf_decision_div,
            'svm_decision_script': svm_decision_script,
            'svm_decision_div': svm_decision_div,
            'script': script,
            'div': div,
            'admission_prediction': admission_prediction,
            'admission_decision_accuracy': admission_decision_accuracy,
            'university_name': query
        }

    elif len(db1) < 25 and len(db1) > 0:
        context = {
                'insufficient_data': 'No enough data for prediction',
                'university_name': query
                 }
    else:
        context = {
            'no_major_available': intended_major,
            'university_name': query
        }

    return render(request, 'decision.html', context)


def college_ranking(request):
    csv_filepathname = "/home/raj/PycharmProjects/university_recommendation/" \
                       "database/dataset_for_clustering.csv"

    names = ['University', 'acceptance_rate', 'average_gre', 'min_gre', 'max_gre', 'average_gre_verbal',
             'min_gre_verbal', 'max_gre_verbal', 'average_gre_quant', 'min_gre_quant', 'max_gre_quant',
             'average_gpa', 'min_gpa', 'max_gpa', 'average_writing', 'min_writing', 'max_writing',
             'average_toefl', 'min_toefl', 'max_toefl']

    df = pd.read_csv(csv_filepathname, names=names, header=None)

    db = df[['University', 'acceptance_rate']]

    ranking1 = db.sort_values(['acceptance_rate', 'University'], ascending=True)
    ranking2 = ranking1.University

    total_universities_list = []
    for university in ranking2:
        if university not in total_universities_list:
            total_universities_list.append(university)

    paginator = Paginator(total_universities_list, 10)  # Show 25 contacts per page
    page = request.GET.get('page')
    try:
        ranking = paginator.page(page)
    except PageNotAnInteger:
        # If page is not an integer, deliver first page.
        ranking = paginator.page(1)
    except EmptyPage:
        # If page is out of range (e.g. 9999), deliver last page of results.
        ranking = paginator.page(paginator.num_pages)

    return render(request, 'ranking.html', {'ranking': ranking, 'range': range(1, 11)})


def random_forest_for_admission_prediction(csv_filepathname, query, intended_major, example):
    names = ['University', 'Major', 'Degree', 'Season', 'class', 'GPA',
             'Verbal', 'Quant', 'AWA', 'TOEFL']

    db = pd.read_csv(csv_filepathname, names=names, header=None)
    db.fillna(0, inplace=True)
    db1 = db[(db['Major'] == intended_major) & (db['University'] == query)]
    db11 = db1
    db12 = db1
    db13 = db1
    db14 = db1
    db15 = db1

    frames1 = [db1, db11, db12, db13, db14, db15]
    db1 = pd.concat(frames1)

    db1['GPA'] = db1.GPA.astype(float)
    db1['Verbal'] = db1.Verbal.astype(float)
    db1['Quant'] = db1.Quant.astype(float)
    db1['AWA'] = db1.AWA.astype(float)
    db1['TOEFL'] = db1.TOEFL.astype(float)
    y = np.array(db1.loc[:, ['class']])
    dataframe = db1.loc[:, 'GPA':'TOEFL']

    X = np.array(dataframe)

    # spliting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

    # Create a random forest classifier. By convention, clf means 'classifier'
    clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                                 max_depth=None, max_features='auto', max_leaf_nodes=None,
                                 min_samples_leaf=1, min_samples_split=2,
                                 min_weight_fraction_leaf=0.0, n_estimators=500, n_jobs=2,
                                 oob_score=False, random_state=None, verbose=0,
                                 warm_start=False)

    # Train the classifier to take the training features and learn how they relate
    # to the training y (the species)
    clf.fit(X_train, y_train)

    prediction = clf.predict(X_test)
    # print (prediction)

    decision_accuracy = clf.predict_proba(X_test)
    # print(decision_accuracy)

    admission_prediction = clf.predict(example)
    print ('--------------------------------------------------')
    print('Prediction_with_random_forest', admission_prediction)
    print ('--------------------------------------------------')

    prediction = clf.predict_proba(example)
    print ('Accuracy_random_forest', prediction)
    print ('--------------------------------------------------')

    # RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
    #                        max_depth=None, max_features='auto', max_leaf_nodes=None,
    #                        min_samples_leaf=1, min_samples_split=2,
    #                        min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=2,
    #                        oob_score=False, random_state=None, verbose=0,
    #                        warm_start=False)
    # clf.predict(X_test)
    # probability = clf.predict_proba(test[features])
    #
    # print (probability)
    #
    # prediction = np.array(clf.predict_proba(example))
    #
    # print ('decision', prediction)

    admission_list = prediction.tolist()
    top_accepted = (admission_list[0])[0]
    top_rejected = (admission_list[0])[1]

    if top_accepted >= 0.9:
        accepted_percent = 88
        rejected_percent = 12
    elif top_rejected >= 0.9:
        accepted_percent = 12
        rejected_percent = 88
    else:
        accepted_percent = top_accepted * 100
        rejected_percent = 100 - top_accepted*100

    plot = figure(plot_width=600, plot_height=450, y_range=(0, 100))

    # plot.vbar(x=[1], width=0.5, bottom=0, top=[100], color=["blue"], alpha=0.7)

    bar1 = plot.vbar(x=[2], width=0.5, bottom=0, top=[accepted_percent],
                     color=["green"], alpha=0.7)

    bar2 = plot.vbar(x=[3], width=0.5, bottom=0, top=[rejected_percent],
                     color=["red"], alpha=0.7)

    plot.yaxis.axis_label = "Percent"

    legend = Legend(items=[
        ("Acceptance Chance", [bar1]),
        ("Rejection Chance", [bar2]),
    ], location=(0, -30))

    plot.add_layout(legend, 'right')

    script, div = components(plot, CDN)

    return script, div


# --------------Support_Vector_Machine_For_admission_prediction------------------
def support_vector_machine(csv_filepathname, query, intended_major, example):

    names = ['University', 'Major', 'Degree', 'Season', 'class', 'GPA',
             'Verbal', 'Quant', 'AWA', 'TOEFL']

    df = pd.read_csv(csv_filepathname, names=names, header=None)

    students_major = intended_major
    university_searched = query

    dataframe = df[df['Major'] == students_major]
    dataframe = dataframe[dataframe['University'] == university_searched]
    db11 = dataframe
    db12 = dataframe
    db13 = dataframe
    db14 = dataframe
    db15 = dataframe

    frames1 = [dataframe, db11, db12, db13, db14, db15]
    dataframe = pd.concat(frames1)

    # dataframe['class'] = [1.0 if x == 'Accepted' else -1.0 for x in dataframe['class']]

    # create design matrix X and target vector y

    dataframe['GPA'] = dataframe.GPA.astype(float)
    dataframe['Verbal'] = dataframe.Verbal.astype(float)
    dataframe['Quant'] = dataframe.Quant.astype(float)
    dataframe['AWA'] = dataframe.AWA.astype(float)
    dataframe['TOEFL'] = dataframe.TOEFL.astype(float)
    y = np.array(dataframe.loc[:, ['class']])
    dataframe = dataframe.loc[:, 'GPA':'TOEFL']

    X = np.array(dataframe)

    # spliting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

    # instantiate learning model
    clf = svm.SVC(kernel='linear', C=0.001)

    clf.fit(X_train, y_train)

    # print ('actual', (y_test))

    # predict
    prediction = clf.predict(X_test)
    # print ('predicted', prediction)
    print ('--------------------------------------------------')

    decision_accuracy = clf.score(X_test, y_test)
    print('Accuracy_with_SVM', decision_accuracy)
    print ('--------------------------------------------------')

    admission_prediction = clf.predict(example)
    print('Prediction_with_SVM', admission_prediction)
    print ('--------------------------------------------------')

    # contengency matrix
    mat = metrics.confusion_matrix(y_test, prediction)
    print(mat)
    tp = mat[0, 0]
    fp = mat[1, 0]
    fn = mat[0, 1]
    tn = mat[1, 1]
    accuracy = float(tp + tn)/( tn + tp + fn + fp)
    print(accuracy)
    print ('--------------------------------------------------')

    if admission_prediction == "Accepted":
        accepted_percent = decision_accuracy*100
        rejected_percent = 100 - (decision_accuracy*100)
    else:
        rejected_percent = decision_accuracy*100

        accepted_percent = 100 - (decision_accuracy*100)

    plot = figure(plot_width=600, plot_height=450, y_range=(0, 100))

    # plot.vbar(x=[1], width=0.5, bottom=0, top=[100], color=["blue"], alpha=0.7)

    bar1 = plot.vbar(x=[2], width=0.5, bottom=0, top=[accepted_percent],
                     color=["green"], alpha=0.7)

    bar2 = plot.vbar(x=[3], width=0.5, bottom=0, top=[rejected_percent],
                     color=["red"], alpha=0.7)

    plot.yaxis.axis_label = "Percent"

    legend = Legend(items=[
        ("Acceptance Chance", [bar1]),
        ("Rejection Chance", [bar2]),
    ], location=(0, -30))

    plot.add_layout(legend, 'right')

    script, div = components(plot, CDN)

    return script, div

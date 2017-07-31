from django.shortcuts import render
import pandas as pd
import numpy as np

# ---University Search function---

def search_university(university_name):
    csv_filepathname = "/home/raj/PycharmProjects/university_recommendation/" \
                       "database/clean238k.csv"
    names = ['University', 'Major', 'Degree', 'Season', 'Decision', 'GPA',
            'Verbal', 'Quant', 'AWA', 'TOEFL', 'Status', 'Comments', 'Comments_Cont']

    df = pd.read_csv(csv_filepathname, names=names, header=None)
    df1 = df[df['University'] == university_name]
    df2 = df1[(df1['Decision'] == 'Accepted')]
    df2['GRE_Scores'] = (df2[['Verbal', 'Quant']].sum(axis=1))
    #df2['GRE_Score'] == df2['Verbal'] + df2['Quant']
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

    total_accepted = df1[df1['Decision'] == 'Accepted'].Decision.count()
    total_rejected = df1[df1['Decision'] == 'Rejected'].Decision.count()
    total_wait_listed = df1[df1['Decision'] == 'Wait listed'].Decision.count()
    total_interview = df1[df1['Decision'] == 'Interview'].Decision.count()
    total_other_decision = df1[df1['Decision'] == 'Other'].Decision.count()

    acceptance_rate = float((total_accepted/float(total_accepted + total_rejected +
                            total_wait_listed + total_interview + total_other_decision))*100)
    majors = df1.Major

    final_majors = []
    for i in majors:
        if i not in final_majors:
            final_majors.append(i)

    total_majors = len(final_majors)
    info_list = [acceptance_rate, average_gre, min_gre, max_gre, average_gre_verbal, min_gre_verbal, 
                max_gre_verbal, average_gre_quant, min_gre_quant, max_gre_quant, average_gpa, min_gpa, 
                max_gpa, average_writing, min_writing, max_writing]
    
    return info_list

# ---Generates Acceptance rate, AVG GRE and GPA of Universities---
def universities_data_generation():
    csv_filepathname = "/home/raj/PycharmProjects/university_recommendation/" \
                       "database/clean238k.csv"

    names = ['Universities', 'Major', 'Degree', 'Season', 'Decision', 'GPA',
             'Verbal', 'Quant', 'AWA', 'TOEFL', 'Status', 'Comments', 'Comments_Cont']

    df = pd.read_csv(csv_filepathname, names=names, header=None)
    df.fillna(0, inplace=True)
    df2 = df[(df['Decision'] == 'Accepted')]

    universities = df2.Universities

    total_universities_list = []
    for university in universities:
        if university not in total_universities_list:
            total_universities_list.append(university)

    record = {}
    for query in total_universities_list:
        record[query] = search_university(query)

        print record

    new_data = pd.DataFrame.from_dict(record, orient='index')
    new_data.to_csv('real_data.csv')
    print (new_data)

    return

#universities_data_generation()









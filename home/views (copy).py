from django.shortcuts import render
import pandas as pd
import numpy as np

# ---University Search function---
def home(request):
    query = request.GET.get('q')

    def search_university(university_name):
        csv_filepathname = "/home/raj/PycharmProjects/university_recommendation/" \
                           "database/clean238k.csv"
        names = ['University', 'Major', 'Degree', 'Season', 'Decision', 'GPA',
                'Verbal', 'Quant', 'AWA', 'TOEFL', 'Status', 'Comments', 'Comments_Cont']

        df = pd.read_csv(csv_filepathname, names=names, header=None)
        df1 = df[df['University'] == university_name]
        df2 = df1[(df1['Decision'] == 'Accepted')]

        average_gre_verbal = (df2['Verbal'].dropna()).mean()
        min_gre_verbal = (df2['Verbal'].dropna()).min()
        max_gre_verbal = (df2['Verbal'].dropna()).max()

        average_gre_quant = (df2['Quant'].dropna()).mean()
        min_gre_quant = (df2['Quant'].dropna()).min()
        max_gre_quant = (df2['Quant'].dropna()).max()

        average_gpa = (df2['GPA'].dropna()).mean()
        min_gpa = (df2['GPA'].dropna()).min()
        max_gpa = (df2['GPA'].dropna()).max()

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
        context = {
            'query': query,
            'acceptance_rate': acceptance_rate,
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
            'max_gpa': max_gpa
        }
        return context

    context = search_university(query)

    return render(request, 'home.html', context)






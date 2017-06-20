from django.shortcuts import render
import pandas as pd


# --University Search function(not working)--
def home(request):
    query = request.GET.get('q')
    print (query)

    csv_filepathname = "/home/raj/PycharmProjects/university_recommendation/" \
                       "database/Final_College_Datasets.csv"
    names = ['University_Name', 'Major', 'Degree', 'Season', 'Decision', 'Undergrad_GPA',
             'GRE_Verbal', 'GRE_Quant', 'GRE_AWA', 'Status']

    df = pd.read_csv(csv_filepathname, names=names, header=None)
    df1 = df[df['University_Name'] == query]
    print (df1)

    total_accepted = df1[df1['Decision'] == 'Accepted'].Decision.count()
    total_rejected = df1[df1['Decision'] == 'Rejected'].Decision.count()
    total_majors = df1.Major.count()
    Majors = df1.Major
    detail = [query, total_accepted, total_rejected, total_majors, Majors]

    list = []
    for i in df1:
        if i not in list:
            list.append(i)
    #print list

    context = {'detail': detail}

    return render(request, 'home.html', context)




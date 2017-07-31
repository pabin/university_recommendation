import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')


def graph_plot():
    csv_filepathname = "/home/raj/PycharmProjects/university_recommendation/" \
                       "database/clean70k.csv"

    names = ['University', 'Major', 'Degree', 'Season', 'Decision', 'GPA',
             'Verbal', 'Quant', 'AWA', 'TOEFL', 'Status', 'Comments', 'Comments_Cont']
    df2 = pd.read_csv(csv_filepathname, names=names, header=None)

    def retrieve_data(university_name):
        df1 = df2[(df2['University'] == university_name) & (df2['Decision'] == 'Accepted')]
        df3 = df1.head(120)

        print (df1['Decision'].value_counts(dropna=False))
        x = np.array(df3.ix[:, 5:6])
        y = np.array(df3.ix[:, 6:7])

        return x, y

    colors1 = ("red")
    colors2 = ("green")

    x, y = retrieve_data('Stanford University')
    Stanford_University = plt.scatter(x, y, s=90, c=colors1, alpha=0.90, marker='o')

    x, y = retrieve_data('Lamar University')
    Lamar_University = plt.scatter(x, y, s=90, c=colors2, alpha=0.90, marker='v', label='Line 2')

    plt.legend([Stanford_University, Lamar_University], ["Stanford University", "Lamar University"],
               loc="upper left")

    plt.ylim(ymin=130, ymax=175)
    plt.xlim(xmin=2, xmax=4.2)

    plt.xlabel('GPA')
    plt.ylabel('GRE_Verbal')
    plt.title('Variation of GPA & GRE Verbal Score among Universities')
    plt.show()
    return


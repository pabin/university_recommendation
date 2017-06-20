# Full path and name to your csv file
csv_filepathname = "/home/raj/PycharmProjects/major_project/database/Final_University_without_oldgre.csv"

# Full path to your django project directory
your_djangoproject_home = "/home/raj/PycharmProjects/university_recommendation"

import sys, os
sys.path.append(your_djangoproject_home)
os.environ['DJANGO_SETTINGS_MODULE'] = 'university.settings'

from college.models import final_college_database

import csv

dataReader = csv.reader(open(csv_filepathname), delimiter=',', quotechar='"')

for row in dataReader:
    if row[0] != 'ZIPCODE':  # Ignore the header row, import everything else
        all_data = final_college_database()
        #all_data.Row_Id = row[0]
        #all_data.Row_Id2 = row[1]
        all_data.University_Name = row[0]
        all_data.Major = row[1]
        all_data.Degree = row[2]
        all_data.Season = row[3]
        all_data.Decision = row[4]
        #all_data.Decision_Method = row[5]
        #all_data.Decision_Date = row[8]
        #all_data.Decision_Timestamp = row[9]
        all_data.Undergrad_GPA = row[5]
        all_data.GRE_Verbal = row[6]
        all_data.GRE_Quant = row[7]
        all_data.GRE_AWA = row[8]
        #all_data.Is_New_GRE = row[14]
        #all_data.GRE_Sub_Test_Score = row[15]
        all_data.Status = row[9]
       #all_data.Post_Data = row[17]
        #all_data.Post_Timestamp = row[18]
        #all_data.Comments = row[19]

        all_data.save()
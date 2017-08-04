
from django.contrib.auth.models import User
from django import forms
from accounts.models import student_info


class UserForm(forms.ModelForm):
    password = forms.CharField(widget=forms.PasswordInput())

    class Meta:
        model = User
        fields = ['username', 'email', 'password', 'first_name', 'last_name']


# --Form that adds extra fields on default system's User fields--
class StudentForm(forms.ModelForm):
    class Meta:
        model = student_info
        fields = ['Intended_Major', 'UnderGrad_GPA', 'GRE_Verbal_Score'
                  , 'GRE_Quant_Score', 'GRE_AWA_Score', 'TOEFL_Score', 'Degree_applying', 'Student_Status']
        help_texts = {'UnderGrad_GPA': "Scale: 1-4 | Eg: 3.40",
                      'GRE_Verbal_Score': "130-170 | Eg: 166",
                      'GRE_Quant_Score': "130-170 | Eg: 168",
                      'GRE_AWA_Score': "1-6 | Eg: 5",
                      'TOEFL_Score': "0-120 | Eg: 104",
                      'Student_Status': "International, American",
                      'Degree_applying': "MS, PhD",
                      'Intended_Major': "Eg: Computer Engineering",

                      }



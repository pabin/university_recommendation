
from django.contrib.auth.models import User
from django import forms
from accounts.models import student_info
from django.contrib.auth.forms import UserChangeForm


class UserForm(forms.ModelForm):
    password = forms.CharField(widget=forms.PasswordInput())

    class Meta:
        model = User
        fields = ['username', 'email', 'password']


# --Form that adds extra fields on default system's User fields--
class StudentForm(forms.ModelForm):
    class Meta:
        model = student_info
        fields = ['First_Name', 'Last_Name', 'Intended_Major', 'UnderGrad_GPA', 'GRE_Verbal_Score'
                  , 'GRE_Quant_Score', 'GRE_AWA_Score', 'TOEFL_Score', 'Student_Status', 'Degree_applying',
                  'List_of_Projects', 'Internships', 'Work_Experience', 'Paper_Publications']


# --Form to edit User Profile--
class EditProfileForm(UserChangeForm):
    class Meta:
        model = student_info
        fields = ['Intended_Major', 'UnderGrad_GPA', 'GRE_Verbal_Score'
                  , 'GRE_Quant_Score', 'GRE_AWA_Score', 'TOEFL_Score', 'Student_Status', 'Degree_applying',
                  'List_of_Projects', 'Internships', 'Work_Experience', 'Paper_Publications']


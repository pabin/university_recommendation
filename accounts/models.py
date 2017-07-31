from django.db import models
from django.contrib.auth.models import User


# --Database Model for Student's Information--
class student_info(models.Model):
    user = models.OneToOneField(User, on_delete= models.CASCADE)
    First_Name = models.CharField(max_length=500)
    Last_Name = models.CharField(max_length=500)
    Intended_Major = models.CharField(max_length=500)
    UnderGrad_GPA = models.CharField(max_length=500)
    GRE_Verbal_Score = models.CharField(max_length=500)
    GRE_Quant_Score = models.CharField(max_length=500)
    GRE_AWA_Score = models.CharField(max_length=500)
    TOEFL_Score = models.CharField(max_length=500)
    Degree_applying = models.CharField(max_length=500)
    Student_Status = models.CharField(max_length=500)
    List_of_Projects = models.TextField()
    Internships = models.TextField()
    Work_Experience = models.TextField()
    Paper_Publications = models.TextField()
    Universities_Added = models.TextField()

    def __str__(self):
        return self.First_Name + ' ' + self.Last_Name + ' - ' + self.Intended_Major


User.profile = property(lambda u: student_info.object.get_or_create(user=u)[0])


class users_university(models.Model):
    user = models.ForeignKey(User, default=1)
    university_name = models.CharField(max_length=250)

    def __str__(self):
        return self.university_name

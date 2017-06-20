from django.db import models
from django.contrib.auth.models import User


# --Database Model for Student's Information--
class student_info(models.Model):
    user = models.OneToOneField(User)
    First_Name = models.CharField(max_length=500)
    Last_Name = models.CharField(max_length=500)
    Intended_Major = models.CharField(max_length=500)
    UnderGrad_GPA = models.CharField(max_length=500)
    GRE_Verbal_Score = models.CharField(max_length=500)
    GRE_Quant_Score = models.CharField(max_length=500)
    GRE_AWA_Score = models.CharField(max_length=500)
    TOEFL_Score = models.CharField(max_length=500, default='NO')
    List_of_Projects = models.CharField(max_length=500, default='NO')
    Internships = models.CharField(max_length=500, default='NO')
    Work_Experience = models.CharField(max_length=500, default='NO')
    Paper_Publications = models.CharField(max_length=500, default='NO')
    Recommendation_Letter1 = models.CharField(max_length=500, default='NO')
    Recommendation_Letter2 = models.CharField(max_length=500, default='NO')
    Student_Status = models.CharField(max_length=500)

    def __str__(self):
        return self.First_Name + ' ' + self.Last_Name + ' - ' + self.Intended_Major


User.profile = property(lambda u: student_info.object.get_or_create(user=u)[0])

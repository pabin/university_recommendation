# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='student_info',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('Username', models.CharField(max_length=500)),
                ('Email', models.CharField(max_length=500)),
                ('Password', models.CharField(max_length=500)),
                ('First_Name', models.CharField(max_length=500)),
                ('Last_Name', models.CharField(max_length=500)),
                ('Intended_Major', models.CharField(max_length=500)),
                ('UnderGrad_GPA', models.CharField(max_length=500)),
                ('GRE_Verbal_Score', models.CharField(max_length=500)),
                ('GRE_Quant_Score', models.CharField(max_length=500)),
                ('GRE_AWA_Score', models.CharField(max_length=500)),
                ('Student_Status', models.CharField(max_length=500)),
            ],
        ),
        migrations.DeleteModel(
            name='student_database',
        ),
    ]

# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='student_database',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
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
    ]

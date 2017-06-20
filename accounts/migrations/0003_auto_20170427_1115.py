# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models
from django.conf import settings


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('accounts', '0002_auto_20170427_0734'),
    ]

    operations = [
        migrations.CreateModel(
            name='student_database',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('Intended_Major', models.CharField(max_length=500)),
                ('UnderGrad_GPA', models.CharField(max_length=500)),
                ('GRE_Verbal_Score', models.CharField(max_length=500)),
                ('GRE_Quant_Score', models.CharField(max_length=500)),
                ('GRE_AWA_Score', models.CharField(max_length=500)),
                ('Student_Status', models.CharField(max_length=500)),
                ('user', models.OneToOneField(to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.DeleteModel(
            name='student_info',
        ),
    ]

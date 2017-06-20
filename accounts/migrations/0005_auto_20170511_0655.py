# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0004_auto_20170427_1139'),
    ]

    operations = [
        migrations.AddField(
            model_name='student_info',
            name='Internships',
            field=models.CharField(default=b'NO', max_length=500),
        ),
        migrations.AddField(
            model_name='student_info',
            name='List_of_Projects',
            field=models.CharField(default=b'NO', max_length=500),
        ),
        migrations.AddField(
            model_name='student_info',
            name='Paper_Publications',
            field=models.CharField(default=b'NO', max_length=500),
        ),
        migrations.AddField(
            model_name='student_info',
            name='Recommendation_Letter1',
            field=models.CharField(default=b'NO', max_length=500),
        ),
        migrations.AddField(
            model_name='student_info',
            name='Recommendation_Letter2',
            field=models.CharField(default=b'NO', max_length=500),
        ),
        migrations.AddField(
            model_name='student_info',
            name='TOEFL_Score',
            field=models.CharField(default=b'NO', max_length=500),
        ),
    ]

# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0006_student_info_work_experience'),
    ]

    operations = [
        migrations.AddField(
            model_name='student_info',
            name='Statement_of_Purpose',
            field=models.TextField(default=b'NO'),
        ),
        migrations.AlterField(
            model_name='student_info',
            name='Internships',
            field=models.TextField(default=b'NO'),
        ),
        migrations.AlterField(
            model_name='student_info',
            name='List_of_Projects',
            field=models.TextField(default=b'NO'),
        ),
        migrations.AlterField(
            model_name='student_info',
            name='Paper_Publications',
            field=models.TextField(default=b'NO'),
        ),
        migrations.AlterField(
            model_name='student_info',
            name='Recommendation_Letter1',
            field=models.TextField(default=b'NO'),
        ),
        migrations.AlterField(
            model_name='student_info',
            name='Recommendation_Letter2',
            field=models.TextField(default=b'NO'),
        ),
        migrations.AlterField(
            model_name='student_info',
            name='Work_Experience',
            field=models.TextField(default=b'NO'),
        ),
    ]

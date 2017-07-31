# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0008_student_info_degree_applying'),
    ]

    operations = [
        migrations.AddField(
            model_name='student_info',
            name='Universities_Added',
            field=models.TextField(default=b'NO'),
        ),
    ]

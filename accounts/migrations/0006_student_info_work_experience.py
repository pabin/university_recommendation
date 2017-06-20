# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0005_auto_20170511_0655'),
    ]

    operations = [
        migrations.AddField(
            model_name='student_info',
            name='Work_Experience',
            field=models.CharField(default=b'NO', max_length=500),
        ),
    ]

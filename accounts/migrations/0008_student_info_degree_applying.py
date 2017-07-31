# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0007_auto_20170622_0557'),
    ]

    operations = [
        migrations.AddField(
            model_name='student_info',
            name='Degree_applying',
            field=models.CharField(default=b'NO', max_length=500),
        ),
    ]

# -*- coding: utf-8 -*-
# Generated by Django 1.11.3 on 2017-08-04 06:10
from __future__ import unicode_literals

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0011_auto_20170729_1333'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='student_info',
            name='Internships',
        ),
        migrations.RemoveField(
            model_name='student_info',
            name='List_of_Projects',
        ),
        migrations.RemoveField(
            model_name='student_info',
            name='Paper_Publications',
        ),
        migrations.RemoveField(
            model_name='student_info',
            name='Universities_Added',
        ),
        migrations.RemoveField(
            model_name='student_info',
            name='Work_Experience',
        ),
    ]
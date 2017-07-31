from django.conf.urls import url
from . import views

app_name = 'accounts'

urlpatterns = [
    url(r'^register/$', views.register, name='register'),
    url(r'^login/$', views.login_user, name='login_user'),
    url(r'^logout/$', views.logout_user, name='logout_user'),
    url(r'^profile/$', views.display_student_profile, name='display_student_profile'),
    url(r'^edit/$', views.edit_profile, name='edit_profile'),
    url(r'^dashboard/$', views.student_dashboard, name='student_dashboard'),
    url(r'^about/$', views.about_us, name='about_us'),
    url(r'^(?P<university_id>[0-9]+)/deleted$', views.delete_dashboard_university, name='delete_dashboard_university'),
]

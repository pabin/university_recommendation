from django.conf.urls import url
from . import views

app_name = 'accounts'

urlpatterns = [
    url(r'^register/$', views.register, name='register'),
    url(r'^login/$', views.login_user, name='login_user'),
    url(r'^logout/$', views.logout_user, name='logout_user'),
    url(r'^profile/$', views.display_student_profile, name='display_student_profile'),
    url(r'^edit/$', views.edit_profile, name='edit_profile'),

]

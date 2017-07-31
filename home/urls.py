from django.conf.urls import url
from . import views


app_name = 'home'

urlpatterns = [
     url(r'^$', views.home, name='home'),
     url(r'^prediction/$', views.admission_prediction_using_knn, name='admission_prediction_using_knn'),
]



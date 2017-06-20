from django.conf.urls import url
from . import views

app_name = 'recommender'

urlpatterns = [
    url(r'^knn/$', views.knn_model, name='knn_model'),
]

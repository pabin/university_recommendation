from django.conf.urls import url
from . import views
import home.views


app_name = 'recommender'

urlpatterns = [
    url(r'^knn/$', views.knn_model, name='knn_model'),

    # using views imported from another app
    url(r'^decision/$', home.views.admission_prediction_using_knn, name='admission_prediction_using_knn'),
    url(r'^ranking/$', home.views.college_ranking, name='college_ranking'),

]

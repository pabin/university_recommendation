"""university URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.8/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Add an import:  from blog import urls as blog_urls
    2. Add a URL to urlpatterns:  url(r'^blog/', include(blog_urls))
"""

from django.conf import settings
from django.conf.urls.static import static
from django.conf.urls import url, include
from django.contrib import admin
from home import views as home_views
#from contact import views as contact_views
#from student import views as student_views


urlpatterns = [
    url(r'^admin/', include(admin.site.urls)),
    #url(r'^$', home_views.home, name='home'),
    url(r'^$', include('home.urls', namespace="home")),

   # url(r'^contact/$', contact_views.contact, name='contact'),
  #  url(r'^student/$', student_views.student, name='student'),
   # url(r'^accounts/', include('allauth.urls')),
    url(r'^accounts/', include('accounts.urls', namespace="accounts")),
    url(r'^recommender/', include('recommender.urls', namespace="recommender")),
   # url(r'^student/', include('student.urls', namespace="student")),
    #url(r'^college/', include('college.urls', namespace="college")),

]


if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path("predict/", views.PredictAPIView.as_view()),  
    path("predictrank/", views.PredictRanksAPIView.as_view())  
]
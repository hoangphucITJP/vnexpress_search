from django.urls import path
from django.views.generic.base import RedirectView

from . import views

urlpatterns = [
    path('', RedirectView.as_view(url='index')),
    path('index', views.index, name='index'),
    path('document', views.document, name='document'),
]

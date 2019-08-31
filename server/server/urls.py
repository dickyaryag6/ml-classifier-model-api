from django.contrib import admin
from django.urls import path
from rest_framework.urlpatterns import format_suffix_patterns

from classifierModel import views

urlpatterns = [
    path('admin/', admin.site.urls),
    # Adding a new URL
    path('model/', views.call_model.as_view())

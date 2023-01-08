"""DjangoWebcamStreaming URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.conf.urls import url , include
from django.contrib import admin
from django.http import StreamingHttpResponse
from django.urls import path
from django.http import HttpResponse
from django.conf import settings

from . import views

from camera import VideoCamera, gen, gen2, gen0

urlpatterns = [
    path('home/', views.homeView , name = 'homeScreen'),
    path('checkup/', views.checkupView, name='checkup'),
    path('results/', views.resultsView, name='result'),
    path('gen/', lambda r: StreamingHttpResponse(gen0(VideoCamera()),content_type='multipart/x-mixed-replace; boundary=frame')),
    path('cam/', lambda r: StreamingHttpResponse(gen(VideoCamera()),content_type='multipart/x-mixed-replace; boundary=frame')),
    path('pulse/', lambda r: StreamingHttpResponse(gen2(VideoCamera()),content_type='multipart/x-mixed-replace; boundary=frame')),
    path('contact/', views.contactView, name="contact"),
    path('about/', views.aboutView, name="about"),
    path('admin/', admin.site.urls),
]

# static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
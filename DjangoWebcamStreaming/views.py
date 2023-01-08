from django.shortcuts import get_object_or_404, render
from django.http import HttpResponse, StreamingHttpResponse

from camera import VideoCamera, gen

def homeView(request):
    template = 'index.html'
    return render(request,template)

def checkupView(request):
    template = 'checkup.html'
    return render(request,template)

def resultView(request):
    template = 'results.html'
    return render(request, template)

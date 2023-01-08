from django.shortcuts import get_object_or_404, render
from django.http import HttpResponse, StreamingHttpResponse

from camera import VideoCamera, gen

def homeView(request):
    template = 'index.html'
    return render(request,template)

def checkupView(request):
    template = 'checkup.html'
    return render(request,template)

def resultsView(request):
    template = 'results.html'
    context = {}
    return render(request,template)

def contactView(request):
    template = 'contact.html'
    return render(request,template)

def aboutView(request):
    return httpResponse()


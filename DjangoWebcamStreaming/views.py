from django.shortcuts import get_object_or_404, render
from django.http import HttpResponse, StreamingHttpResponse

from camera import VideoCamera, gen

def homeView(request):
    template = 'index.html'
    return render(request,template)

def checkupView(request):
    template = 'checkup.html'
    return render(request,template)

def resultsView(request,gen,age,h,arml,legl,waistl,w):
    template = 'results.html'
    cont = {'gen':gen, 'age':age, 'h':h, 'arml':arml, "legl":legl, "waistl":waistl, "w":w}
    print(cont)
    return render(request,template,context=cont)

def results2View(request,gen,abc,cd):
    template = 'results.html'
    cont = {'gen':gen,'abc':abc,'cd':cd}
    print("hii")
    print(cont)
    return render(request,template,context=cont)

def contactView(request):
    template = 'contact.html'
    return render(request,template)

def aboutView(request):
    return httpResponse()


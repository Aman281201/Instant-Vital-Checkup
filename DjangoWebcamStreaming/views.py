from django.shortcuts import get_object_or_404, render
from django.http import HttpResponse


def homeView(request):
    template = 'index.html'
    return render(request,template)



# def detail(request, question_id):
#     return HTTPResponse("You're looking at question %s." % question_id)

# def results(request, question_id):
#     response = "You're looking at the results of question %s."
#     return HttpResponse(response % question_id)

# def vote(request, question_id):
#     return HttpResponse("You're voting on question %s." % question_id)

# def index(request):
#     latest_question_list = Question.objects.order_by('-pub_date')[:5]
#     output = ', '.join([q.question_text for q in latest_question_list])
#     return HttpResponse(output)

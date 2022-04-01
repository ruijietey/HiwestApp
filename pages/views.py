from django.shortcuts import render
from django.http import HttpResponse
from src.westsum import summarize


# Create your views here.
def home_view(request, *args, **kwargs):
    summary_context = {}
    if request.method == 'POST':
        input_data = request.POST['text'].strip()
        num_sents = int(request.POST['numSents'])
        print(request.POST.getlist('model'))
        model_selected = request.POST.getlist('model')[0]
        sentences, summary_ids = summarize(input_data, num_sents, model_selected)
        summary = ""
        for id in summary_ids:
            summary += sentences[id]['text']
            summary += " "
        summary_context = {
        "sents": sentences,
        "summary": summary,
        "summary_ids": summary_ids,
        }
    return render(request, "home.html", summary_context)
        # print(request.GET)

        
    


def extract_summary(request, *args, **kwargs):
    # return HttpResponse("<h1>Hello World</h1>")
    return render(request, "home.html", {})


def contact_view(request, *args, **kwargs):
    return render(request, "contact.html", {})


def about_view(request, *args, **kwargs):
    # return HttpResponse("<h1>Hello World</h1>")
    return render(request, "about.html", {})
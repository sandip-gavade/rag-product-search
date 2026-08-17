from django.urls import path

from .views import RAGAnswerView

urlpatterns = [
    path("answer/", RAGAnswerView.as_view(), name="rag-answer"),
]

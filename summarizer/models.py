from django.db import models

# Create your models here.
class Summarizer(models.Model):
    title       = models.TextField()
    description = models.TextField()
    input_data  = models.PositiveSmallIntegerField()
    summary     = models.TextField()

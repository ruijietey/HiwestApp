# HiwestApp

This is repository for the Django web application using [HiWestSum](https://github.com/ruijietey/HiWestSum). 
Please setup a virtual environment with the packages listed below:
<ul>
  <li>transformers==4.17.0</li>
  <li>sentencepiece==0.1.96</li>
  <li>Django==4.0.3</li>
  <li>nltk==3.7</li>
  <li>tokenizers==0.11.6</li>
</ul>

## Checkpoints
Please download the checkpoints on https://drive.google.com/drive/folders/1Vi5eaT-7Us-6OmLoWaCPRRTZsWL2OGDz?usp=sharing and move them into ./checkpoints directory.

## Running the application
Remember to use the correct virtual environment with packages installed. Use the following command at root directory for Django application:
```
python manage.py runserver
```
Then open the application on http://127.0.0.1:8000/

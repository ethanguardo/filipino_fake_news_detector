FILIPINO FAKE NEWS DETECTOR 

Summary: 
  Content includes train.py that builds the machine learning tool.
  This tool is used to detect credible and non-credible news articles.

Specification:
  Uses TF-idf and a Naive Bayes classifier 
  Trained on the fake news corpus (~22,000 articles)
  Assignes a confidence score.

Installation And Setup:

Clone Repository:
  git clone https://github.com/ethanguardo/filipino_fake_news_detector.git
  
Install Dependencies:
  pip install -r requirements.txt

Train Model (if joblib not installed):
  python3 train.py

Run Classifier:
  python3 classify.py


  

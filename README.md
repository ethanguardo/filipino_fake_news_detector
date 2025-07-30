**Filipino Fake News Detector**

**Summary**

This project includes a machine learning tool that detects **credible** and **non-credible** Philippine news articles.  
- `train.py` – Builds and trains the model.  
- `classify.py` – Uses the trained model to classify new articles with a **confidence score**.

**Specifications**
- Uses **TF-IDF** for feature extraction and a **Naive Bayes** classifier.
- Trained on the **Philippine Fake News Corpus** (~22,000 articles).
- Outputs a **credibility prediction** with a probability score.


**Installation and Setup**

**Clone Repository:**
```bash
git clone https://github.com/ethanguardo/filipino_fake_news_detector.git
cd filipino_fake_news_detector

```

**Install dependencies:**
```bash
  pip install -r requirements.txt
```

**Train Model (if joblib not installed):**
```bash
  python3 train.py
```


**Run Classifier:**
```bash
  python3 classify.py
```



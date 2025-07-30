**Filipino Fake News Detector**

**Summary**

This project includes a machine learning tool that detects **credible** and **non-credible** Philippine news articles.  
- `train.py` – Builds and trains the model from the given dataset.  
- `classify.py` – Uses the trained model to classify new articles with a **confidence score**.

**Dataset**

This project uses the **Philippine Fake News Corpus** created by Aaron Carl Fernandez.

- GitHub Repository: [https://github.com/aaroncarlfernandez/FakeNewsPhilippines](https://github.com/aaroncarlfernandez/FakeNewsPhilippines)
- Research Paper: [Computing the Linguistic-Based Cues of Credible and Not Credible News in the Philippines: Towards Fake News Detection](https://www.researchgate.net/publication/334724915_Computing_the_Linguistic-Based_Cues_of_Credible_and_Not_Credible_News_in_the_Philippines_Towards_Fake_News_Detection)

Dataset from 2019 by Aaron Carl Fernandez. Used for fair academic use for research purposes.

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



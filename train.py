import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

# Load dataset
df = pd.read_csv("Philippine Fake News Corpus.csv")

# preprocessing - headline and content are combined into one column 
df["text"] = df["Headline"].fillna('') + " " + df["Content"].fillna('')
df = df[df["text"].str.strip() != ""]  

# Creates a new column called label and assignes 0 or 1 depending on credibilitiy
df["label"] = df["Label"].map({"Credible": 0, "Not Credible": 1})

# Split the dataset into training and testing sets
# - Uses sklearn.model_selection.train_test_split() to separate features and labels.
#   Function docs: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
#   Source code (GitHub): https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/model_selection/_split.py
# - stratify=df["label"]: Ensures the proportion of credible and not credible articles 
#   is preserved in both training and test sets (maintains class balance).
#   Reference: https://stackoverflow.com/questions/34842405/how-to-use-stratify-option-in-sklearn-train-test-split
# - test_size=0.2: Allocates 20% of the data for testing and 80% for training.
#   Reference: https://stackoverflow.com/questions/38250710/how-to-split-data-into-training-and-testing-sets-in-scikit-learn
# - random_state=42: Sets a fixed seed for reproducibility (same split every run).
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# Some parts of the pipeline were generated with the assistance of an AI tool.
# line 41-44, 52, 61-62, 65-66 was reviewed, tested, and modified by the author to fit the project's needs.

# - Uses sklearn.pipeline.Pipeline to chain preprocessing and modeling steps together.
# - TfidfVectorizer() converts text into TF-IDF features with unigrams and bigrams.
# - MultinomialNB() is chosen for its efficiency and suitability for text classification.
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=50000, min_df=2, stop_words="english")),
    ("clf", MultinomialNB(class_prior=[0.5, 0.5]))
])

# Train the model on the training set.
# - pipeline.fit() runs the entire Pipeline:
#   • Applies TF-IDF vectorization on X_train to convert text to numerical features.
#   • Fits the Multinomial Naive Bayes classifier on the transformed data.
# - This is where the model learns patterns to distinguish between credible and non-credible articles.
# Docs: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline.fit
pipeline.fit(X_train, y_train)

#  Evaluate the trained model using the testing set.
# - pipeline.predict(X_test) applies the same TF-IDF transformation to X_test and predicts labels using the trained Naive Bayes classifier.
# - classification_report() provides precision, recall, F1-score, and support for each class (Credible and Not Credible).
# - This step measures how well the model generalizes to unseen data, similar to testing an intrusion detection system on new traffic.
# Docs:
#   • Pipeline.predict(): https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline.predict
#   • classification_report(): https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred, target_names=["Credible", "Not Credible"]))

# Saves the model
joblib.dump(pipeline, "filipino_fake_news_model.joblib")
print("Model saved as filipino_fake_news_model.joblib")

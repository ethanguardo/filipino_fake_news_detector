import joblib

# Load the trained model
model = joblib.load("filipino_fake_news_model.joblib")

# Your input (new article)
headline = input("Enter the headline of the article: ")
content = input("Enter the content of the article: ")

# Combine them just like during training
text = headline + " " + content

# Predicts whether credible or not 
# Uses scikit-learn's Pipeline.predict() to classify the article (0 = Credible, 1 = Not Credible)
# Ref: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
prediction = model.predict([text])[0]

# Gives probabilities from Naive Bayes for confidence scoring
# Ref: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
probability = model.predict_proba([text])[0]

# Output result
label = "Credible" if prediction == 0 else "Not Credible"

# Select label and highest probability as confidence score
# Naive Bayes outputs different probabilities, so max(probability)
# represents the model's confidence in its prediction (Zhang, 2004).
confidence = round(max(probability) * 100, 2)

print(f"Prediction: {label} (Confidence: {confidence}%)")

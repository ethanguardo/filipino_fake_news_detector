import joblib

# Load the trained model
model = joblib.load("filipino_fake_news_model.joblib")

# 2. Your input (new article)
headline = input("Enter the headline of the article: ")
content = input("Enter the content of the article: ")


# 3. Combine them just like during training
text = headline + " " + content

# 4. Predict
prediction = model.predict([text])[0]
probability = model.predict_proba([text])[0]

# 5. Output result
label = "Credible" if prediction == 0 else "Not Credible"
confidence = round(max(probability) * 100, 2)

print(f"Prediction: {label} (Confidence: {confidence}%)")

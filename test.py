import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Загрузка модели и TF-IDF векторизатора
loaded_model = joblib.load('model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

text = "TV hostess Lazareva faces up to 7 years in prison for justifying terrorism" # Тестовый текст

# Преобразование текста в числовые признаки с помощью TF-IDF векторизатора
tfidf_text = tfidf_vectorizer.transform([text])

# Предсказание метки класса
prediction = loaded_model.predict(tfidf_text)

# Получение значений решающей функции
decision_values = loaded_model.decision_function(tfidf_text)

# Преобразование значений решающей функции в вероятности
probabilities = np.exp(decision_values) / np.sum(np.exp(decision_values))

# Получение вероятности класса "фейк"
fake_probability = probabilities[0]

# Вычисление уверенности в процентах
confidence = fake_probability * 100

# Вывод результатов
print(f'Эта новость: {prediction[0]}, {confidence:.2f}%')
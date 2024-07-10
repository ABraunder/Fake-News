import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

# Загрузка данных

data = pd.read_csv('fake_news.csv')

# Разделение данных на признаки и метки
X = data['text']
y = data['label']

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Преобразование текстовых данных в числовые признаки с помощью TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

# Построение модели PassiveAggressiveClassifier
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

# Предсказание на тестовых данных
y_pred = pac.predict(tfidf_test)

# Оценка точности модели
score = accuracy_score(y_test, y_pred)
print(f'Точность: {round(score*100, 2)}%')

# Построение матрицы ошибок
conf_matrix = confusion_matrix(y_test, y_pred)

# Функция для визуализации матрицы ошибок
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Матрица ошибок',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Нормированная матрица")
    else:
        print('Матрица без нормирования')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Правильно')
    plt.xlabel('Прогноз')
    plt.tight_layout()

# Визуализация матрицы ошибок
plt.figure(figsize=(10, 7))
plot_confusion_matrix(conf_matrix, classes=['FAKE', 'REAL'], title='Матрица ошибок')
plt.show()

# Отчет о классификации
print(classification_report(y_test, y_pred, target_names=['FAKE', 'REAL']))

# Дополнительные визуализации
# Построение графика распределения предсказанных меток
sns.countplot(y_pred)
plt.title('Распределение предсказанных меток')
plt.xlabel('Метка')
plt.ylabel('Количество')
plt.show()

import joblib

joblib.dump(pac, 'model.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')


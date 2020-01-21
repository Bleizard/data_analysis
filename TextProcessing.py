#!/usr/bin/env python
# coding: utf-8
import nltk
import numpy as np
from TextHelpers import tokenize_ru, tf_idf, hal_matrix, get_texts_from_wiki, make_plot

nltk.download('punkt')
nltk.download('stopwords')

TOP_TERMS_COUNT = 20
text_names = [
    "doc",
    "bakery",
    "cereals_macarons",
    "dish",
    "drinks",
    "fish",
    "meat",
    "salads",
    "sauses",
    "soups",
    "vegetables_mushrooms"
    ]

TEXTS_COUNT = len(text_names)

print("Частотный анализ")
with open('texts/doc.txt', 'r') as file:
    content = file.read().replace('\n', ' ')
print("Токенизация текста")
tokens = tokenize_ru(content)

print("Подсчёт частот")
freq_dist = nltk.FreqDist(w.lower() for w in tokens)

print(f"Топ {TOP_TERMS_COUNT} результатов")
print(freq_dist.most_common(TOP_TERMS_COUNT))

wiki_texts = get_texts_from_wiki(20)
print("Проведение TF-IDF анализа")
with open('texts/doc.txt', 'r') as file:
    content = file.read().replace('\n', ' ')
    top_terms = tf_idf(content, wiki_texts)[:TOP_TERMS_COUNT]
    print(f'Результат TF-IDF. Первые {TOP_TERMS_COUNT} результатов')
    print(top_terms)


make_plot(freq_dist, 50, "frequency_plot.png")

print("Построение HAL матриц")
with open('texts/doc.txt', 'r') as file:
    content = file.read().replace('\n', ' ')
    print("Токенизируем весь текст")
    text_dict = list(set(tokenize_ru(content)))


hal_matrices = dict()
for text in text_names[1:]:
    print(f'Построение HAL матрицы для документа: {text}')
    with open(f'texts/{text}.txt', 'r') as file:
        content = file.read().replace('\n', ' ')
        tokens = tokenize_ru(content)
        hal_matrices[text] = hal_matrix(tokens, text_dict, 5)

hal_matrices_copy = hal_matrices
results = dict()
for m_n in text_names[1:]:
    res = []
    for i in range(TOP_TERMS_COUNT):
        r, c = np.unravel_index(hal_matrices_copy[m_n].argmax(), hal_matrices_copy[m_n].shape)
        res.append(tokens[r] + ' ' + tokens[c])
        hal_matrices_copy[m_n][(r, c)] = -1
    results[m_n] = res

print(f'Топ {TOP_TERMS_COUNT}словосочетаний для каждой темы')
print(results)
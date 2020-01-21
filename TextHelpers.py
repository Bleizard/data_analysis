import re
import string
import numpy as np
from stop_words import get_stop_words
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from corus import load_wiki
import pymorphy2
import matplotlib.pyplot as plt


def tokenize_ru(file_text):
    # firstly let's apply nltk tokenization
    file_text = re.sub(r"$\d+\W+|\b\d+\b|\W+\d+$", " ", file_text)
    file_text = re.sub(r'[^\w\s]', '', file_text)

    tokens = word_tokenize(file_text)

    # let's delete punctuation symbols
    tokens = [i for i in tokens if (i not in string.punctuation)]

    # deleting stop_words
    stop_words = stopwords.words('russian') + get_stop_words('ru')
    stop_words.extend(['в', '', '•', '—', '–', 'к', 'на', '№', '©', '►','3–4','1–2','2–3', '5–7', '15–20', '10-15', '20–25', '½', '...'])
    tokens = [i for i in tokens if ((i not in stop_words) and (len(i)>1) and (not i.isnumeric()))]
    # cleaning words
    tokens = [i.replace("«", "").replace("»", "") for i in tokens]

    morph = pymorphy2.MorphAnalyzer()
    morphs = []
    for item in tokens:
        morphs.append(sorted(morph.parse(item), key=lambda p: p.score)[0].normal_form)

    return morphs


# ## TF IDF
def tf_idf(corpus, wiki_texts):
    corpus = [corpus]+wiki_texts
    vectorizer = TfidfVectorizer(tokenizer=tokenize_ru)
    X = vectorizer.fit_transform(corpus)
    features = np.array(vectorizer.get_feature_names())
    return features[(-X.toarray()[0]).argsort()]


def hal_matrix(tokenized_text, all_tokens, frame_len=5):
    m_size = len(all_tokens)
    hal = np.zeros([m_size, m_size])
    for t in range(len(tokenized_text) - frame_len):
        t_token = tokenized_text[t]
        for f_off in range(1, frame_len):
            off_token = tokenized_text[t + f_off]
            r = all_tokens.index(t_token)
            c = all_tokens.index(off_token)
            hal[r, c] += frame_len - f_off
    return hal


def get_texts_from_wiki(sample_size):
    print("Чтение текстов из Вики")
    path = 'ruwiki-latest-pages-articles.xml.bz2'
    records = load_wiki(path)
    print("Подготовка текстов из Вики")
    wiki_texts = []
    for i in range(sample_size):
        wiki_texts.append(next(records).text)
    print("Подготовка текстов из Вики завершена")
    return wiki_texts


def make_plot(frequency_dist, COMMONS_COUNTER, filename):
    frequency_dist = frequency_dist.most_common(COMMONS_COUNTER)
    indices = np.arange(len(frequency_dist))
    plt.figure(figsize=(15, 10))
    plt.bar(indices, list(map(lambda x: x[1], frequency_dist)))
    plt.xticks(indices, list(map(lambda x: x[0], frequency_dist)), rotation='vertical')
    plt.tight_layout()
    plt.savefig(filename)
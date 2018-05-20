# -*- coding=GBK -*-
"""
extractSentenceWords

"""
import unicodedata
import sys
import re
import itertools
import logging
from gensim.models.phrases import Phrases
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=logging.INFO)

from nltk.stem.wordnet import WordNetLemmatizer
reload(sys)
sys.setdefaultencoding('utf8')

unicode_punc_tbl = dict.fromkeys( i for i in xrange(128, sys.maxunicode)
                      if unicodedata.category(unichr(i)).startswith('P') )


# def lemmatize(word):
#     if word in skip_words:
#         return word
#     n_word = WordNetLemmatizer().lemmatize(word, 'v')
#     if n_word not in special_words:
#         temp_token = WordNetLemmatizer().lemmatize(n_word, 'n')
#         if (n_word != temp_token and not re.search(r'ss$', n_word)):
#             n_word = temp_token
#     return n_word

def extractSentenceWords(doc, remove_url=True, remove_punc="utf-8", min_length=1, lemma=False, sent=True, replace_digit=False):
    if remove_punc:
        # ensure doc_u is in unicode
        if not isinstance(doc, unicode):
            encoding = remove_punc
            doc_u = doc.decode(encoding)
        else:
            doc_u = doc
        # remove unicode punctuation marks, keep ascii punctuation marks
        doc_u = doc_u.translate(unicode_punc_tbl)
        if not isinstance(doc, unicode):
            doc = doc_u.encode(encoding)
        else:
            doc = doc_u

    if remove_url:
        re_url = r"(https?:\/\/)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)"
        doc = re.sub( re_url, "", doc )

    sentences = re.split( r"\s*[;:`\"()?!{}]\s*|--+|\s*-\s+|''|\.\s|\.$|\.\.+|��|��", doc )  # comment comma
    wc = 0
    wordsInSentences = []
    wnl = WordNetLemmatizer()
    for sentence in sentences:
        if sentence == "":
            continue

        if not re.search( "[A-Za-z0-9]", sentence ):
            continue

        # words = re.split( r"\s+\+|^\+|\+?[\-*\/&%=_<>\[\]~\|\@\$]+\+?|\'\s+|\'s\s+|\'s$|\s+\'|^\'|\'$|\$|\\|\s+", sentence )
        words = re.split(r"[\s+,\-*\/&%=_<>\[\]~\|\@\$\\]",
                         sentence)
        words = filter( lambda w: w, words)
        words = map(lambda w: w.lower(), words)
        if replace_digit:
            map(lambda w: re.sub(r'\d+', '<digit>', w), words)
        if lemma:
            words = map(lambda w: wnl.lemmatize(w, 'v'), words)
        if len(words) >= min_length:
            wordsInSentences.append(words)
            wc += len(words)
    if not sent:
        return list(itertools.chain.from_iterable(wordsInSentences)), wc
    return wordsInSentences, wc



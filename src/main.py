"""
Online LDA with a soft alignment to integrate previous states.
"""

import os
import gc
import resource
import logging
import itertools
import time
import json
import cPickle
import re
import numpy as np
from scipy import sparse
from scipy.stats import entropy
from datetime import datetime, timedelta
import nltk
from nltk.corpus import stopwords
from gensim import corpora
from collections import defaultdict
from gensim.models import Word2Vec, LdaMulticore, TfidfModel
from extractSentenceWords import *
from onlineLDA import *
from config import Config
from extract_phrase import extract_phrases

bigram = None; trigram = None; wv_model = None
my_stoplst = ["app", "good", "excellent", "awesome", "please", "they", "very", "too", "like", "love", "nice", "yeah",
"amazing", "lovely", "perfect", "much", "bad", "best", "yup", "suck", "super", "thank", "great", "really",
"omg", "gud", "yes", "cool", "fine", "hello", "alright", "poor", "plz", "pls", "google", "facebook",
"three", "ones", "one", "two", "five", "four", "old", "new", "asap", "version", "times", "update", "star", "first",
"rid", "bit", "annoying", "beautiful", "dear", "master", "evernote", "per", "line", "oh", "ah", "cannot", "doesnt",
"won't", "dont", "unless", "you're", "aren't", "i'd", "can't", "wouldn't", "around", "i've", "i'll", "gonna", "ago",
"you'll", "you'd", "28th", "gen", "it'll", "vice", "would've", "wasn't", "year", "boy", "they'd", "isnt", "1st", "i'm",
"nobody", "youtube", "isn't", "don't", "2016", "2017", "since", "near", "god"]


# dataset
app_files = Config.get_datasets()
app_files_pre = {}
validate_files = Config.get_validate_files()
candidate_num = Config.get_candidate_num()
topic_num = Config.get_topic_num()
win_size = Config.get_window_size()
bigram_min = Config.get_bigram_min()
trigram_min = Config.get_trigram_min()
info_num = Config.get_info_num()
store_num = Config.get_store_num()
val_index = Config.get_validate_or_not()


def extract_review():
    """
    Extract reviews with time and version stamp
    :return:
    """
    timed_reviews = {}
    num_docs = 0
    num_words = 0
    for apk, app in app_files:
        timed_reviews[apk] = []
        with open(app) as fin:
            lines = fin.readlines()
        for l_id, line in enumerate(lines):
            line = line.strip()
            terms = line.split("******")
            if len(terms) != info_num:
                logging.error("review format error at %s in %s" % (apk, line))
                continue
            if not store_num: ## for ios
                date = terms[3]
                version = terms[4]
            else:             ## for android
                date = terms[2]
                version = terms[3]
            review_o = terms[1]
            review_p, wc = extractSentenceWords(review_o)
            review = list(build_phrase(review_p))
            review = [list(replace_digit(s)) for s in review]
            rate = float(terms[0]) if re.match(r'\d*\.?\d+', terms[0]) else 2.0  # 2.0 is the average rate
            timed_reviews[apk].append({"review": review, "date": date, "rate": rate, "version": version})
            num_docs += 1
            num_words += wc
            if l_id % 1000 == 0:
                logging.info("processed %d docs of %s" % (l_id, apk))
    logging.info("total read %d reviews, %d words."%(num_docs, num_words))
    return timed_reviews

def replace_digit(sent):
    for w in sent:
        if w.isdigit():
            yield '<digit>'
        else:
            yield w

def build_phrase(doc):
    # load phrase model
    return trigram[bigram[doc]]


def update_phrase():
    """
    Update bigram and trigram model
    :return:
    """
    for apk, app in app_files:
        with open(app) as fin:
            lines = fin.readlines()
        for line in lines:
            line = line.strip()
            terms = line.split("******")
            if len(terms) != info_num:
                logging.error("review format error at %s in %s" % (apk, line))
                continue
            review_o = terms[1]
            review_p, wc = extractSentenceWords(review_o)
            bigram.add_vocab(review_p)
            trigram.add_vocab(bigram[review_p])
        # update
        bigram.save("../model/bigram.model")
        trigram.save("../model/trigram.model")

def load_phrase():
    global bigram
    global trigram
    bigram = Phrases.load(os.path.join("..", "model", "bigram.model"))
    trigram = Phrases.load(os.path.join("..", "model", "trigram.model"))

def save_obj(filename, rst):
    with open(filename, 'w') as fout:
        cPickle.dump(rst, fout)
def load_obj(filename):
    with open(filename) as fin:
        return cPickle.load(fin)

def build_AOLDA_input_version(timed_reviews=None):
    """
    build version-aligned input for AOLDA
    :param timed_reviews:
    :return:
    """
    if timed_reviews is None:
        with open("../result/timed_reviews") as fin:
            timed_reviews = json.load(fin)
    stoplist = stopwords.words('english') + my_stoplst

    OLDA_input = {}
    for apk, reviews in timed_reviews.items():
        # build a dictionary to store the version and review
        version_dict = {}
        input = []
        rate = []
        tag = []
        for review in reviews:
            review_ver = review['version']
            if review_ver == "Unknown":
                continue
            if review_ver not in version_dict:
                version_dict[review_ver] = ([], [])
            version_dict[review_ver][0].append(review['review'])
            version_dict[review_ver][1].append(review['rate'])

        # re-arrange the version sequence
        for ver in sorted(version_dict.iterkeys(), key=lambda s: map(int, s.split('.'))):
            if len(version_dict[ver][0]) > 50:          # skip versions with not enough reviews
                tag.append(ver)
                input.append(version_dict[ver][0])
                rate.append(version_dict[ver][1])

        dict_input = list(itertools.chain.from_iterable(list(itertools.chain.from_iterable(input))))
        dictionary = corpora.Dictionary(dict_input)
        dictionary.filter_tokens(map(dictionary.token2id.get, stoplist))
        dictionary.compactify()
        dictionary.filter_extremes(no_below=2, keep_n=None)
        dictionary.compactify()

        # for each interval, build bow
        input_X = []
        for t_i, text_period in enumerate(input):
            # construct sparse matrix
            text_period = list(itertools.chain.from_iterable(text_period))      # sentence level to doc level
            row = []
            col = []
            value = []
            r_id = 0
            for k, text in enumerate(text_period):
                empty = True
                for i, j in dictionary.doc2bow(text):
                    row.append(r_id)
                    col.append(i)
                    value.append(j)
                    empty = False
                if not empty:
                    r_id += 1
            input_X.append(sparse.coo_matrix((value, (row, col)), shape=(r_id, len(dictionary))))
        OLDA_input[apk] = (dictionary, input_X, input, rate, tag)      # input: raw input, with time and sent
    return OLDA_input

def generate_labeling_candidates(OLDA_input):
    """
    Filter phrase labels and choose for candidates
    :param OLDA_input:
    :return:
    """
    phrases = {}
    for apk, item in OLDA_input.items():
        dic, _, _1, _2, _3= item
        phrases[apk] = defaultdict(int)
        # filter bigram and trigram
        for word in dic.values():
            if '_' in word:
                phrase = word
                words, tags = zip(*nltk.pos_tag(phrase.split(b'_')))
                match = False
                for tag in tags:
                    if re.match(r"^NN", tag):
                        match = True
                        continue
                    if re.match(r"DT", tag):
                        match = False
                        break
                    if re.match(r"RB", tag):
                        match = False
                        break
                for word in words:
                    if word in stopwords.words('english') + my_stoplst:     # remove stop word
                        match = False
                        break
                    if len(word) < 3:
                        match = False
                        break
                    if "\\'" in word:
                        match = False
                        break
                if match:
                    # keep phrase
                    phrases[apk][phrase] = 1
    return phrases

def OLDA_fit(OLDA_input, n_topics, win_size):
    phis = {}
    theta = {}
    for apk, item in OLDA_input.items():
        dictionary, input_X, _, _1, _2 = item
        olda_model = OLDA(n_topics=n_topics, n_iter=1000, refresh=500, window_size=win_size)
        olda_model.fit(input_X)
        phis[apk] = olda_model.B
        theta[apk] = olda_model.A
        fout = open("../result/topic_words_%s_%s_%s"%(apk, n_topics, win_size), 'w')
        for t_i, phi in enumerate(phis[apk]):
            fout.write("time slice %s\n"%t_i)
            for i, topic_dist in enumerate(phi):
                topic_words = [dictionary[w_id] for w_id in np.argsort(topic_dist)[:-10:-1]]
                fout.write('Topic {}: {}\n'.format(i, ' '.join(topic_words)))
            fout.write('\n')
        fout.close()
    return phis

def count_occurence(dic, rawinput, label_ids):
    count = []
    for d_i, rawinput_i in enumerate(rawinput):
        count_i = defaultdict(int)
        for input in list(itertools.chain.from_iterable(rawinput_i)):
            bow = dic.doc2bow(input)
            for id, value in bow:
                count_i[id] += value
                if id in label_ids[d_i]:
                    for idx, valuex in bow:
                        count_i[id, idx] += min(value, valuex)    # label always first
        count.append(count_i)
    return count

def total_count_(dic, rawinput):
    total_count = []
    for rawinput_i in rawinput:
        total_count_i = 0
        for input in list(itertools.chain.from_iterable(rawinput_i)):
            bow = dic.doc2bow(input)
            for id, value in bow:
                total_count_i += value
        total_count.append(total_count_i)
    return total_count

def get_candidate_label_ids(dic, labels, rawinput):
    all_label_ids = map(dic.token2id.get, labels)
    label_ids = []

    for rawinput_i in rawinput:
        count = defaultdict(int)
        for input in list(itertools.chain.from_iterable(rawinput_i)):
            bow = dic.doc2bow(input)
            for id, value in bow:
                if id in all_label_ids:
                    count[id] += value
        label_ids.append(count.keys())
    return label_ids

def get_candidate_sentences_ids(rawinput, rates):
    sent_ids = []
    sent_rates = []
    index = 0
    for t_i, rawinput_i in enumerate(rawinput):
        sent_id = []
        sent_rate = []
        for i_d, input_d in enumerate(rawinput_i):
            for i_s, input_s in enumerate(input_d):
                if len(input_s) < 5:          # length should be bigger than 5
                    continue
                sent_id.append(index + i_s)
                sent_rate.append(rates[t_i][i_d])
            index += len(input_d)
        sent_ids.append(sent_id)
        sent_rates.append(sent_rate)
    return sent_ids, sent_rates

def get_sensitivities(dic, rawinput, rates, label_ids):
    sensi = []
    for t_i, rawinput_i in enumerate(rawinput):
        sensi_t = []
        label_sensi = [[] for _ in label_ids[t_i]]
        for d_i, input in enumerate(rawinput_i):
            doc_input = list(itertools.chain.from_iterable(input))
            bow = dic.doc2bow(doc_input)
            for id, value in bow:
                if id in label_ids[t_i]:
                    label_sensi[label_ids[t_i].index(id)].append([rates[t_i][d_i], len(doc_input)]) # record the rate and length
        for rl in label_sensi:
            rl = np.array(rl)
            m_rl = np.mean(rl, 0)
            sensi_t.append(np.exp(- m_rl[0]/np.log(1+m_rl[1])))
        sensi.append(np.array(sensi_t))
    return sensi

def get_sensitivities_sent(rawinput_sent, sent_rates, sent_ids):
    sensi = []
    for t_i, sent_id in enumerate(sent_ids):
        sensi_i = []
        for id, s_id in enumerate(sent_id):
            r = sent_rates[t_i][id]
            l = len(rawinput_sent[s_id])
            sensi_i.append(np.exp(- r / float(np.log(l))))
        sensi.append(np.array(sensi_i))
    return sensi


def JSD(P, Q):
    """
    Jensen-Shannon divergence
    :param P:
    :param Q:
    :return:
    """
    _M = 0.5 * (P + Q)
    return 0.5 * (entropy(P, _M) + entropy(Q, _M))

def sim_topic_word(phi, label_id, count):
    # sim = 0
    c_l = np.array([np.log((count[label_id, w_id] + 1) / float((count[w_id] + 1) * (count[label_id] + 1))) for w_id in range(len(phi))])
    return np.dot(phi, c_l)

def topic_labeling(OLDA_input, apk_phis, phrases, mu, lam, theta, save=True):
    """
    Topic labeling for phrase and sentence
    :param OLDA_input:
    :param apk_phis:
    :param phrases:
    :param mu:
    :param lam:
    :param theta:
    :param save:
    :return:
    """
    logging.info("labeling topics(mu: %f, lam: %f, theta: %f)......" % (mu, lam, theta))
    apk_jsds = {}
    for apk, item in OLDA_input.items():
        dictionary, _, rawinput, rates, tag = item
        phis = apk_phis[apk]
        labels = phrases[apk].keys()
        # label_ids = map(dictionary.token2id.get, labels)
        label_ids = get_candidate_label_ids(dictionary, labels, rawinput)
        count = count_occurence(dictionary, rawinput, label_ids)
        total_count = total_count_(dictionary, rawinput)
        sensi_label = get_sensitivities(dictionary, rawinput, rates, label_ids)
        rawinput_sent = list(itertools.chain.from_iterable(list(itertools.chain.from_iterable(rawinput))))
        sent_ids, sent_rates = get_candidate_sentences_ids(rawinput, rates)
        sensi_sent = get_sensitivities_sent(rawinput_sent, sent_rates, sent_ids)
        jsds = []
        label_phrases = []; label_sents = []; emerge_phrases = []; emerge_sents = []
        if save:
            result_path="../result/%s"%apk
            if not os.path.exists(result_path):
                os.makedirs(result_path)
            fout_labels = open(os.path.join(result_path, "topic_labels"), 'w')
            fout_emerging = open(os.path.join(result_path, "emerging_topic_labels"), 'w')
            fout_sents = open(os.path.join(result_path, "topic_sents"), "w")
            fout_emerging_sent = open(os.path.join(result_path, "emerging_topic_sents"), 'w')
            fout_topic_width = open(os.path.join(result_path, "topic_width"), 'w')

        for t_i, phi in enumerate(phis):
            # label topic
            logging.info("labeling topic at %s slice of %s" % (t_i, apk))
            topic_label_scores = topic_labeling_(count[t_i], total_count[t_i], label_ids[t_i], sensi_label[t_i], phi, mu, lam)
            topic_label_sent_score = topic_label_sent(dictionary, phi, rawinput_sent, sent_ids[t_i], sensi_sent[t_i], mu, lam)

            # write to file: topic phrase
            if save:
                fout_labels.write("time slice %s, tag: %s\n"%(t_i, tag[t_i]))
                for tp_i, label_scores in enumerate(topic_label_scores):
                    fout_labels.write("Topic %d:"%tp_i)
                    for w_id in np.argsort(label_scores)[:-candidate_num-1:-1]:
                        fout_labels.write("%s\t%f\t" % (dictionary[label_ids[t_i][w_id]], label_scores[w_id]))
                    fout_labels.write('\n')

                fout_sents.write("time slice %s, tag: %s\n" % (t_i, tag[t_i]))
                for tp_i, sent_scores in enumerate(topic_label_sent_score):
                    fout_sents.write("Topic %d:"%tp_i)
                    for s_id in np.argsort(sent_scores)[:-candidate_num-1:-1]:
                        fout_sents.write("%s\t%f\t"%(" ".join(rawinput_sent[sent_ids[t_i][s_id]]), sent_scores[s_id]))
                    fout_sents.write('\n')

            # store for verification
            label_phrases_ver = []; label_sents_ver = []
            for tp_i, label_scores in enumerate(topic_label_scores):
                label_phrases_ver.append(
                    [dictionary[label_ids[t_i][w_id]] for w_id in np.argsort(label_scores)[:-candidate_num-1:-1]])
            label_phrases.append(list(itertools.chain.from_iterable(label_phrases_ver)))
            for tp_i, sent_scores in enumerate(topic_label_sent_score):
                label_sents_ver.append(
                    [rawinput_sent[sent_ids[t_i][s_id]] for s_id in np.argsort(sent_scores)[:-candidate_num-1:-1]])
            label_sents.append(list(itertools.chain.from_iterable(label_sents_ver)))

            # detect emerging topic
            logging.info("detecting topic at %s slice of %s" % (t_i, apk))
            if save and t_i == 0:
                topic_width = count_width(dictionary, label_phrases_ver, count[t_i], sensi_label[t_i], label_ids[t_i])
                for theta in topic_width:
                    fout_topic_width.write("%f\t" % theta)
                fout_topic_width.write("\n")
                continue   # skip the first epoch
            emerging_label_scores, emerging_sent_scores = topic_detect(rawinput_sent, dictionary, phi, phis[t_i-1], count[t_i], count[t_i-1], total_count[t_i],
                                                 total_count[t_i-1], label_ids[t_i], sent_ids[t_i], sensi_label[t_i], sensi_sent[t_i], jsds, theta, mu, lam)
            # write to file
            if save:
                fout_emerging.write("time slice %s, tag: %s\n"%(t_i, tag[t_i]))
                for tp_i, label_scores in enumerate(emerging_label_scores):
                    fout_emerging.write("Topic %d: "%tp_i)
                    if np.sum(label_scores) == 0:
                        fout_emerging.write('None\n')
                    else:
                        for w_id in np.argsort(label_scores)[:-4:-1]:
                            fout_emerging.write("%s\t%f\t" % (dictionary[label_ids[t_i][w_id]], label_scores[w_id]))
                        fout_emerging.write('\n')
                fout_emerging_sent.write("time slice %s, tag: %s\n"%(t_i, tag[t_i]))
                for tp_i, sent_scores in enumerate(emerging_sent_scores):
                    fout_emerging_sent.write("Topic %d: "%tp_i)
                    if np.sum(sent_scores) == 0:
                        fout_emerging_sent.write('None\n')
                    else:
                        for s_id in np.argsort(sent_scores)[:-4:-1]:
                            fout_emerging_sent.write("%s\t%f\t" % (" ".join(rawinput_sent[sent_ids[t_i][s_id]]), sent_scores[s_id]))
                        fout_emerging_sent.write('\n')
            # store for verification
            emerge_phrases_ver = []; emerge_sents_ver = []
            emerge_phrases_width_ver = []
            for tp_i, label_scores in enumerate(emerging_label_scores):
                if np.sum(label_scores) == 0:
                    emerge_phrases_width_ver.append([])
                    continue
                emerge_phrases_ver.append(
                    [dictionary[label_ids[t_i][w_id]] for w_id in np.argsort(label_scores)[:-4:-1]])
                emerge_phrases_width_ver.append(
                    [dictionary[label_ids[t_i][w_id]] for w_id in np.argsort(label_scores)[:-4:-1]])
            emerge_phrases.append(emerge_phrases_ver)
            # merge emerge to label
            label_emerge_ver = [set(l)|set(e) for l, e in zip(label_phrases_ver, emerge_phrases_width_ver)]
            topic_width = count_width(dictionary, label_emerge_ver, count[t_i], sensi_label[t_i], label_ids[t_i])
            for tp_i, sent_scores in enumerate(emerging_sent_scores):
                if np.sum(sent_scores) == 0:
                    continue
                emerge_sents_ver.append(
                    [rawinput_sent[sent_ids[t_i][s_id]] for s_id in np.argsort(sent_scores)[:-4:-1]])
            emerge_sents.append(emerge_sents_ver)
            # write topic width
            if save:
                for theta in topic_width:
                    fout_topic_width.write("%f\t" % theta)
                fout_topic_width.write("\n")

        ############################################
        if val_index:
            validation(validate_files[apk], label_phrases, label_sents, emerge_phrases, emerge_sents)
        ############################################

        if save:
            fout_labels.close()
            fout_sents.close()
            fout_emerging.close()
            fout_emerging_sent.close()
            fout_topic_width.close()
        apk_jsds[apk] = jsds
    return apk_jsds

def topic_labeling_(count, total_count, label_ids, sensi, phi, mu, lam):
    topic_label_scores = rank_topic_label(count, total_count, phi, label_ids, mu)
    topic_label_scores += lam * sensi
    return topic_label_scores

# rank the label according to similarity of the topic dist and divergence of other topic dist
def rank_topic_label(count, total_count, phi, label_ids, mu=0.2):
    # matrix implementation for speed-up
    # construct topic matrix
    mu_div = mu / (len(phi) - 1)
    c_phi = phi * (1 + mu_div) - np.sum(phi, 0) * mu_div
    # construct label count matrix
    c_label_m = np.empty((len(label_ids), len(phi[0])), dtype=float)
    for ind, label_id in enumerate(label_ids):
        for w_id in range(len(phi[0])):
            c_label_m[ind, w_id] = count.get((label_id, w_id)) * total_count / float((count.get(w_id) + 1) * (count.get(label_id) + 1)) if (label_id, w_id) in count else 1.0
    c_label_m = np.log(c_label_m)
    # compute score matrix
    topic_label_scores = np.dot(c_phi, np.transpose(c_label_m))
    return topic_label_scores

def topic_detect(rawinput_sents, dic, phi, last_phi, count, last_count, total_count, last_total_count, label_ids, sent_ids, sensi_label, sensi_sent, jsds, theta, mu, lam):
    # matrix implementation for speed-up
    # construct count label matrix
    c_label_m = np.empty((len(label_ids), len(phi[0])), dtype=float)
    c_last_label_m = np.empty((len(label_ids), len(phi[0])), dtype=float)
    for ind, label_id in enumerate(label_ids):
        for w_id in range(len(phi[0])):
            c_label_m[ind, w_id] = count.get((label_id, w_id)) * total_count / float((count.get(w_id) + 1) * (count.get(w_id) + 1)) if (label_id, w_id) in count else 1.0
            c_last_label_m[ind, w_id] = last_count.get((label_id, w_id)) * last_total_count / float((last_count.get(w_id) + 1) * (last_count.get(label_id) + 1)) if (label_id, w_id) in last_count else 1.0

    c_label_m = np.log(c_label_m)
    c_last_label_m = np.log(c_last_label_m)
    # construct sentence count matrix
    sent_count = np.empty((len(sent_ids), len(phi[0])), dtype=float)
    for ind, s_id in enumerate(sent_ids):
        bow = dic.doc2bow(rawinput_sents[s_id])
        len_s = len(rawinput_sents[s_id])
        for w_id in range(len(phi[0])):
            sent_count[ind, w_id] = 0.00001
        for k, v in bow:
            sent_count[ind, k] = v / float(len_s)
    # # construct residuals
    # phi_logphi = np.log(phi) * phi
    # phi_logphi_last = np.log(last_phi) * last_phi

    # read topic distribution \phi
    emerging_label_scores_rst = np.zeros((len(phi), len(label_ids)))
    emerging_sent_scores_rst = np.zeros((len(phi), len(sent_ids)))
    js_d = []
    for t_i, phi_i in enumerate(phi):
        # labeling
        js_divergence = JSD(phi_i, last_phi[t_i])
        js_d.append(js_divergence)
        jsds.append(js_divergence)
        # logging.info("JSD for phi is %f"%js_divergence)
    # compute mean and variance of jsds
    js_mean = np.mean(jsds[:-3*len(phi)-1:-1])
    js_std = np.std(jsds[:-3*len(phi)-1:-1])
    # logging.info("JSD threshold is %f"%(js_mean+1.25*js_std))
    emerging_index = np.array(js_d) > js_mean + 1.25*js_std
    # TOPIC DETECT
    phi_e = phi[emerging_index]
    phi_last_e = last_phi[emerging_index]
    E = float(np.sum(emerging_index))
    if E == 0:
        return emerging_label_scores_rst, emerging_sent_scores_rst
    # TOPIC DETECT: construct phi - last_phi
    phi_m = (1 + mu/E) * phi_e - theta * last_phi[emerging_index] - mu/E * np.sum(phi_e, 0)
    # TOPIC DETECT: construct residuals
    residuals_m = (1 + mu/E) * np.log(phi_e) * phi_e - theta * np.log(phi_last_e) * phi_last_e - mu/E * np.sum(np.log(phi_e) * phi_e, 0)
    # TOPIC DETECT: compute labels
    emerging_label_scores = np.dot((1 + mu/E) * phi_e - mu/E * np.sum(phi_e, 0), np.transpose(c_label_m)) - theta * np.dot(last_phi[emerging_index], np.transpose(c_last_label_m)) + lam * sensi_label
    emerging_sent_scores = np.dot(phi_m, np.transpose(np.log(sent_count))) - np.sum(residuals_m, 1, keepdims=True) + lam * sensi_sent

    emerging_label_scores_rst[emerging_index] = emerging_label_scores
    emerging_sent_scores_rst[emerging_index] = emerging_sent_scores

    return emerging_label_scores_rst, emerging_sent_scores_rst

# rank sentence representation for topic
def topic_label_sent(dic, phi, rawinput_sents, sent_ids, sensi, mu, lam):
    # construct topic matrix
    mu_div = mu / (len(phi) - 1)
    c_phi = phi * (1 + mu_div) - np.sum(phi, 0) * mu_div
    # construct residual
    phi_logphi = phi * np.log(phi)
    residual_1 = mu_div * np.sum(phi_logphi)        # residual_1 is a value
    residual_2 = (1 + mu_div) * np.sum(phi_logphi, 1, keepdims=True)    # residual_2 is a n_topic*1
    # construct sentence count matrix
    sent_count = np.empty((len(sent_ids), len(phi[0])), dtype=float)
    for ind, s_id in enumerate(sent_ids):
        bow = dic.doc2bow(rawinput_sents[s_id])
        len_s = len(rawinput_sents[s_id])
        for w_id in range(len(phi[0])):
            sent_count[ind, w_id] = 0.00001
        for k, v in bow:
            sent_count[ind, k] = v / float(len_s)

    phi_sent = np.dot(c_phi, np.transpose(np.log(sent_count))) + residual_1 - residual_2 + lam * sensi
    return phi_sent

def count_width(dictionary, label_phrases_ver, counts, sensi_labels, label_ids):
    count_width_rst = []
    for phrases in label_phrases_ver:
        t_count = 0
        for phrase in phrases:
            pid = dictionary.token2id.get(phrase)
            t_count += np.log(counts.get(pid)+1) * sensi_labels[label_ids.index(pid)]
        count_width_rst.append(t_count)
    return np.array(count_width_rst)

def validation(logfile, label_phrases, label_sents, emerge_phrases, emerge_sents):
    # read changelog
    clog = []
    with open(logfile) as fin:
        for line in fin.readlines():
            line = line.strip()
            issue_kw = map(lambda s: s.strip().split(), line.split(","))
            clog.append(issue_kw)
    # check alignment
    if len(clog) != len(label_phrases):
        logging.error("length not corrected: %d, %d"%(len(clog), len(label_phrases)))
        exit(0)
    # compare topic label using keyword
    # load word2vec model
    wv_model = Word2Vec.load(os.path.join("..", "model", "wv", "word2vec_app.model"))
    label_phrase_precisions = []; label_phrase_recalls = []; label_sent_precisions = []; label_sent_recalls = []
    em_phrase_precisions = []; em_phrase_recalls = []; em_sent_precisions = []; em_sent_recalls = []
    # two list: [['keyword1', 'keyword2', ...], ['keyword1', 'keyword2', ...]]
    #           [['label1', 'label2', ...], ['label1', 'label2', ...]]
    for id, ver in enumerate(clog):
        if ver == [[]]: # skip the empty version changelog
            continue
        label_phrase_match_set = set(); label_phrase_issue_match_set = set(); label_sent_match_set = set(); label_sent_issue_match_set = set()
        em_phrase_match_set = set(); em_phrase_issue_match_set = set(); em_sent_match_set = set(); em_sent_issue_match_set = set()

        if id != len(clog) - 1 and clog[id+1] != [[]]:         # merge changelog with next version
            m_ver = ver + clog[id+1]
        else:
            m_ver = ver
        # phrase
        for issue in m_ver:
            for kw in issue:
                kw_match = False
                for w in label_phrases[id]:
                    label_match = False
                    for w_s in w.split("_"):
                        if sim_w(kw, w_s, wv_model) > 0.6:
                            # hit
                            #logging.info("hit: %s -> %s"%(w, kw))
                            label_match = True
                            kw_match = True
                            break
                    if label_match: # if label match found, add label to match set
                        label_phrase_match_set.add(w)
                if kw_match:    # if kw match found, add issue to match set
                    label_phrase_issue_match_set.add("_".join(issue))

        # sentence
        for issue in m_ver:
            for kw in issue:
                kw_match = False
                for sent in label_sents[id]:
                    for w in sent:
                        label_match = False
                        for w_s in w.split("_"):
                            if sim_w(kw, w_s, wv_model) > 0.6:
                                # hit
                                #logging.info("hit: %s -> %s"%(w, kw))
                                label_match = True
                                kw_match = True
                                break
                        if label_match:
                            label_sent_match_set.add("_".join(sent))   # if label match found, skip to next sentence
                            break
                if kw_match:
                    label_sent_issue_match_set.add("_".join(issue))

        # check emerging issue label
        # merge current version and next version
        # if id != len(clog) - 1:
        #     m_ver = ver + clog[id+1]
        # else:
        #     m_ver = ver
        if id != 0:     # skip the first epoch
            for issue in m_ver:
                for kw in issue:
                    kw_match = False
                    for tws in emerge_phrases[id-1]:
                        for w in tws:
                            label_match = False
                            for w_s in w.split("_"):
                                if sim_w(kw, w_s, wv_model) > 0.6:
                                    # hit
                                    #logging.info("hit: %s -> %s" % (w, kw))
                                    label_match = True
                                    kw_match = True
                                    break
                            if label_match:
                                em_phrase_match_set.add("_".join(tws))
                                break
                    if kw_match:
                        em_phrase_issue_match_set.add("_".join(issue))

            # sentence
            for issue in m_ver:
                for kw in issue:
                    kw_match = False
                    for tsents in emerge_sents[id-1]:
                        sent = list(itertools.chain.from_iterable(tsents))
                        label_match = False
                        for w in sent:
                            for w_s in w.split("_"):
                                if sim_w(kw, w_s, wv_model) > 0.6:
                                    # hit
                                    #logging.info("hit: %s -> %s" % (w, kw))
                                    label_match = True
                                    kw_match = True
                                    break
                            if label_match:
                                em_sent_match_set.add("_".join(sent))  # if label match found, skip to next sentence
                                break
                    if kw_match:
                        em_sent_issue_match_set.add("_".join(issue))

        # compute
        label_phrase_precision = len(label_phrase_match_set) / float(len(label_phrases[id]))
        label_phrase_recall = len(label_phrase_issue_match_set) / float(len(m_ver))
        label_sent_precision = len(label_sent_match_set) / float(len(label_sents[id]))
        label_sent_recall = len(label_sent_issue_match_set) / float(len(m_ver))
        label_phrase_precisions.append(label_phrase_precision)
        label_phrase_recalls.append(label_phrase_recall)
        label_sent_precisions.append(label_sent_precision)
        label_sent_recalls.append(label_sent_recall)

        if id != 0:
            if len(emerge_phrases[id-1]) != 0:
                em_phrase_precision = len(em_phrase_match_set) / float(len(emerge_phrases[id-1]))
                em_phrase_precisions.append(em_phrase_precision)
            em_phrase_recall = len(em_phrase_issue_match_set) / float(len(ver))
            if len(emerge_sents[id-1]) != 0:
                em_sent_precision = len(em_sent_match_set) / float(len(emerge_sents[id-1]))
                em_sent_precisions.append(em_sent_precision)
            em_sent_recall = len(em_sent_issue_match_set) / float(len(ver))
            em_phrase_recalls.append(em_phrase_recall)
            em_sent_recalls.append(em_sent_recall)
    label_phrase_fscore = 2 * np.mean(label_phrase_recalls) * np.mean(em_phrase_precisions) / (np.mean(label_phrase_recalls) + np.mean(em_phrase_precisions))
    label_sent_fscore = 2 * np.mean(label_sent_recalls) * np.mean(em_sent_precisions) / (np.mean(label_sent_recalls) + np.mean(em_sent_precisions))
    logging.info("Phrase label precision: %s\trecall: %f"%(np.mean(label_phrase_precisions), np.mean(label_phrase_recalls)))
    logging.info("Sentence label precision: %s\trecall: %f" % (np.mean(label_sent_precisions), np.mean(label_sent_recalls)))
    logging.info(
        "Emerging phrase precision: %s\trecall: %f" % (np.mean(em_phrase_precisions), np.mean(em_phrase_recalls)))
    logging.info(
        "Emerging sentence precision: %s\trecall: %f" % (np.mean(em_sent_precisions), np.mean(em_sent_recalls)))
    logging.info("Phrase F1 score: %f"%label_phrase_fscore)
    logging.info("Sentence F1 score: %f" % label_sent_fscore)
    with open("../result/statistics.txt", "a") as fout:
        fout.write("%s\t%f\t%f\t%f\t%f\t%f\t%f\n"%(logfile, np.mean(label_phrase_recalls), np.mean(label_sent_recalls), np.mean(em_phrase_precisions), np.mean(em_sent_precisions), label_phrase_fscore, label_sent_fscore))

def sim_w(w1, w2, wv_model):
    if w1 not in wv_model or w2 not in wv_model:
        return 0.0
    return wv_model.similarity(w1, w2)

def meminfo(str):
    print(str, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000)

def save_phrase(review_path, bigram_num, trigram):
    for apk, app in app_files:
        extract_phrases(app)

if __name__ == '__main__':
    extract_phrases(app_files, bigram_min, trigram_min)
    load_phrase()

    timed_reviews = extract_review()

    OLDA_input = build_AOLDA_input_version(timed_reviews)
    start_t = time.time()
    apk_phis = OLDA_fit(OLDA_input, topic_num, win_size)
    phrases = generate_labeling_candidates(OLDA_input)
    topic_labeling(OLDA_input, apk_phis, phrases, 1.0, 0.75, 0.0, save=True)
    print("Totally takes %.2f seconds" % (time.time() - start_t))

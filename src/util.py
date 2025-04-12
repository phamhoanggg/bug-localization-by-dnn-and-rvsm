import csv
import json
import pickle
import re
import os
import random
import timeit
import string
import numpy as np
from datetime import datetime
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing import SourceFile

from  assets import  stop_words

# def tsv2dict(tsv_path):
#     """ Converts a tab separated values (tsv) file into a list of dictionaries

#     Arguments:
#         tsv_path {string} -- path of the tsv file
#     """
#     reader = csv.DictReader(open(tsv_path, "r"), delimiter="\t")
#     dict_list = []
#     for line in reader:
#         line["files"] = [
#             os.path.normpath(f[0:])
#             for f in line["files"].strip().split()
#             if (f.startswith("java/") or f.startswith("modules/") or f.startswith("test/")) and f.endswith(".java")
#         ]
#         line["raw_text"] = line["summary"] + line["description"]
#         # line["summary"] = clean_and_split(line["summary"][11:])
#         # line["description"] = clean_and_split(line["description"])
#         line["report_time"] = datetime.strptime(
#             line["report_time"], "%Y-%m-%d %H:%M:%S"
#         )
#         # print("Line: ", line)
#         # break

#         dict_list.append(line)
#     return dict_list

def csv2dict_for_br(br_path):
    """ Converts a tab separated values (tsv) file into a list of dictionaries

    Arguments:
        tsv_path {string} -- path of the tsv file
    """
    reader = csv.DictReader(open(br_path, "r"), delimiter=",")
    dict_list = []
    for line in reader:
        line["files"] = [
            os.path.normpath(f.strip()).replace("\\", "/")
            for f in line["fixed_files"].split(";")
            if (
                    f.strip().replace("\\", "/").startswith("java/")
                    or f.strip().replace("\\", "/").startswith("modules/")
                    or f.strip().replace("\\", "/").startswith("test/")
                    or f.strip().replace("\\", "/").startswith("webapps/")
                    # f.strip().replace("\\", "/").startswith("ajde/")
                    # or f.strip().replace("\\", "/").startswith("ajde.core/")
                    # or f.strip().replace("\\", "/").startswith("ajdoc/")
                    # or f.strip().replace("\\", "/").startswith("asm/")
                    # or f.strip().replace("\\", "/").startswith("bcel-builder/")
                    # or f.strip().replace("\\", "/").startswith("bridge/")
                    # or f.strip().replace("\\", "/").startswith("loadtime/")
                    # or f.strip().replace("\\", "/").startswith("org.aspectj.ajdt.core/")
                    # or f.strip().replace("\\", "/").startswith("org.aspectj.matcher/")
                    # or f.strip().replace("\\", "/").startswith("tests/")
                    # or f.strip().replace("\\", "/").startswith("weaver/")
                    and f.strip().endswith(".java")
            )
        ]
        summary_dict = json.loads(line["summary"].replace("'", '"'))
        description_dict = json.loads(line["description"].replace("'", '"'))
        line["raw_text"] = " ".join(summary_dict["stemmed"]) + " " + " ".join(description_dict["stemmed"])
        line["report_time"] = datetime.strptime(
            line["report_time"], "%Y-%m-%d %H:%M:%S"
        )

        dict_list.append(line)
    return dict_list


def csv2dict(csv_path):
    """ Converts a comma separated values (csv) file into a dictionary

    Arguments:
        csv_path {string} -- path to csv file
    """
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f, delimiter=",")
        csv_dict = list()
        for line in reader:
            csv_dict.append(line)

    return csv_dict

def pickle2dict(pickle_path):
    """ Converts a pickle file into a dictionary

    Arguments:
        pickle_path {string} -- path to pickle file
    """
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    return data

def clean_and_split(text):
    """ Remove all punctuation and split text strings into lists of words

    Arguments:
        text {string} -- input text
    """
    table = str.maketrans(dict.fromkeys(string.punctuation))
    clean_text = text.translate(table)
    word_list = [s.strip() for s in clean_text.strip().split()]
    return word_list


def top_k_wrong_files(right_files, br_raw_text, java_files, k=50):
    """ Randomly samples 2*k from all wrong files and returns metrics
        for top k files according to rvsm similarity.

    Arguments:
        right_files {list} -- list of right files
        br_raw_text {string} -- raw text of the bug report
        java_files {dictionary} -- dictionary of source code files

    Keyword Arguments:
        k {integer} -- the number of files to return metrics (default: {50})
    """

    # Randomly sample 2*k files
    #randomly_sampled = random.sample(set(java_files.keys()) - set(right_files), 2 * k)
    randomly_sampled = random.sample(list(set(java_files.keys()) - set(right_files)), 2 * k)

    all_files = []

    for filename in randomly_sampled:
        src = java_files[filename]

        src_text = ' '.join(src.all_content["stemmed"])
        rvsm = cosine_sim(br_raw_text, src_text)
        cns = class_name_similarity(br_raw_text, src)

        all_files.append((filename, rvsm, cns))
        # try:
        #     src = java_files[filename]
        #
        #     rvsm = cosine_sim(br_raw_text, src)
        #     cns = class_name_similarity(br_raw_text, src)
        #
        #     all_files.append((filename, rvsm, cns))
        # except:
        #     pass

    top_k_files = sorted(all_files, key=lambda x: x[1], reverse=True)[:k]

    return top_k_files


def stem_tokens(tokens):
    """ Remove stopword and stem

    Arguments:
        tokens {list} -- tokens to stem 
    """
    stemmer = PorterStemmer()
    removed_stopwords = [
        stemmer.stem(item) for item in tokens if item not in stopwords.words("english")
    ]

    return removed_stopwords


def normalize(text):
    """ Lowercase, remove punctuation, tokenize and stem

    Arguments:
        text {string} -- A text to normalize
    """
    remove_punc_map = dict((ord(char), None) for char in string.punctuation)
    removed_punc = text.lower().translate(remove_punc_map)
    tokenized = word_tokenize(removed_punc)
    stemmed_tokens = stem_tokens(tokenized)

    return stemmed_tokens


def cosine_sim(text1, text2):
    """ Cosine similarity with tfidf

    Arguments:
        text1 {string} -- first text
        text2 {string} -- second text
    """
    vectorizer = TfidfVectorizer(tokenizer=normalize, token_pattern=None, min_df=1, stop_words=list(stop_words))
    tfidf = vectorizer.fit_transform([text1, text2])
    sim = ((tfidf * tfidf.T).toarray())[0, 1]

    return sim


def get_all_source_code(start_dir):
    """ Creates corpus starting from 'start_dir'

    Arguments:
        start_dir {string} -- directory path to start
    """
    files = {}
    start_dir = os.path.normpath(start_dir)
    for dir_, dir_names, file_names in os.walk(start_dir):
        for filename in [f for f in file_names if f.endswith(".java")]:
            src_name = os.path.join(dir_, filename)
            with open(src_name, "r") as src_file:
                src = src_file.read()

            file_key = src_name.split(start_dir)[1]
            file_key = file_key[len(os.sep) :]
            files[file_key] = src

    return files


def get_months_between(d1, d2):
    """ Calculates the number of months between two date strings

    Arguments:
        d1 {datetime} -- date 1
        d2 {datetime} -- date 2
    """

    diff_in_months = abs((d1.year - d2.year) * 12 + d1.month - d2.month)

    return diff_in_months


def most_recent_report(reports):
    """ Returns the most recently submitted previous report that shares a filename with the given bug report

    Arguments:
        filename {string} -- the name of the shared Java file
        current_date {datetime} -- until date
        bug_reports {list of dictionaries} -- list of all bug reports
    """

    if len(reports) > 0:
        return max(reports, key=lambda x: x.get("report_time"))

    return None


def previous_reports(filename, until, bug_reports):
    """ Returns a list of previously filed bug reports that share a file with the current bug report

    Arguments:
        filename {string} -- the name of the shared Java file
        until {datetime} -- until date
        bug_reports {list of dictionaries} -- list of all bug reports
    """
    filename = filename.replace("\\", "/")
    return [
        br
        for br in bug_reports
        if (filename in br["files"] and br["report_time"] < until)
    ]


def bug_fixing_recency(br, prev_reports):
    """ Calculates the Bug Fixing Recency as defined by Lam et al.

    Arguments:
        report1 {dictionary} -- current bug report
        report2 {dictionary} -- most recent bug report
    """
    mrr = most_recent_report(prev_reports)

    if br and mrr:
        return 1 / float(
            get_months_between(br.get("report_time"), mrr.get("report_time")) + 1
        )

    return 0


def collaborative_filtering_score(raw_text, prev_reports):
    """[summary]

    Arguments:
        raw_text {string} -- raw text of the bug report 
        prev_reports {list} -- list of previous reports
    """

    prev_reports_merged_raw_text = ""
    for report in prev_reports:
        prev_reports_merged_raw_text += report["raw_text"]


    cfs = cosine_sim(raw_text, prev_reports_merged_raw_text)
    # print("Raw text: ", raw_text)
    # print("Prev reports merged raw text: ", prev_reports_merged_raw_text)
    # print("CFS: ", cfs)
    return cfs


def class_name_similarity(raw_text, source_line):
    """[summary]

    Arguments:
        raw_text {string} -- raw text of the bug report 
        source_line {string} -- line in src_file.csv
    """
    classname_dict = source_line.class_names
    class_names_text = ' '.join(classname_dict['stemmed'])

    class_name_sim = cosine_sim(raw_text, class_names_text)

    return class_name_sim


def helper_collections(samples, bug_report_csv, only_rvsm=False):
    """ Generates helper function for calculations
    
    Arguments:
        samples {list} -- samples from features_eclipse_base.csv
    
    Keyword Arguments:
        only_rvsm {bool} -- If True only 'rvsm' features are added to 'sample_dict'. (default: {False})
    """
    sample_dict = {}
    for feature_row in samples:
        sample_dict[feature_row["report_id"]] = []

    for feature_row in samples:
        temp_dict = {}

        values = [float(feature_row["rVSM_similarity"])]
        if not only_rvsm:
            values += [
                float(feature_row["collab_filter"]),
                float(feature_row["classname_similarity"]),
                float(feature_row["bug_recency"]),
                float(feature_row["bug_frequency"]),
            ]
        temp_dict[os.path.normpath(feature_row["file"])] = values

        sample_dict[feature_row["report_id"]].append(temp_dict)

    bug_reports = csv2dict(bug_report_csv)
    br2files_dict = {}

    for bug_report in bug_reports:
        br2files_dict[bug_report['id']] = bug_report["fixed_files"]

    return sample_dict, bug_reports, br2files_dict


def topk_accuarcy(test_bug_reports, sample_dict, br2files_dict, clf=None):
    """ Calculates top-k accuracies and returns ranks for MRR/MAP calculation
    
    Arguments:
        test_bug_reports {list of dictionaries} -- list of all bug reports
        sample_dict {dictionary of dictionaries} -- a helper collection for fast accuracy calculation
        br2files_dict {dictionary} -- dictionary for "bug report id - list of all related files in features_eclipse_base.csv" pairs
    
    Keyword Arguments:
        clf {object} -- A classifier with 'predict()' function. If None, rvsm relevancy is used. (default: {None})
    """
    topk_counters = [0] * 20
    negative_total = 0
    ranks = []  # Store ranks for MRR/MAP calculation

    for bug_report in test_bug_reports:
        dnn_input = []
        corresponding_files = []
        bug_id = bug_report["id"]

        try:
            for temp_dict in sample_dict[bug_id]:
                java_file = list(temp_dict.keys())[0]
                features_for_java_file = list(temp_dict.values())[0]
                dnn_input.append(features_for_java_file)
                corresponding_files.append(java_file)
        except Exception as e:
            print("Error: ", e)
            negative_total += 1
            continue

        # for temp_dict in sample_dict[bug_id]:
        #     java_file = list(temp_dict.keys())[0]
        #     features_for_java_file = list(temp_dict.values())[0]
        #     dnn_input.append(features_for_java_file)
        #     corresponding_files.append(java_file)


        # Calculate relevancy for all files
        relevancy_list = []
        if clf:  # dnn classifier
            relevancy_list = clf.predict(dnn_input)
        else:  # rvsm
            relevancy_list = np.array(dnn_input).ravel()

        # Find rank of first correct file
        correct_files = br2files_dict[bug_id]
        found_rank = float('inf')
        
        for i, file in enumerate(corresponding_files):
            if str(file) in correct_files:
                found_rank = i + 1  # Convert to 1-based indexing
                break
                
        if found_rank != float('inf'):
            ranks.append(found_rank)

        # Top-1, top-2 ... top-20 accuracy
        for i in range(1, 21):
            max_indices = np.argpartition(relevancy_list, -i)[-i:]
            for corresponding_file in np.array(corresponding_files)[max_indices]:
                if str(corresponding_file) in br2files_dict[bug_id]:
                    topk_counters[i - 1] += 1
                    break

    acc_dict = {}
    for i, counter in enumerate(topk_counters):
        acc = counter / (len(test_bug_reports) - negative_total)
        acc_dict[i + 1] = round(acc, 3)

    # Calculate MRR and MAP for k=20
    accuracy, mrr, map_score = mrr_map(ranks, 20)
    acc_dict['mrr'] = round(mrr, 3)
    acc_dict['map'] = round(map_score, 3)

    return acc_dict


def mrr_map(ranks, k):
    """ Calculates top-k accuracy, MRR and MAP

    Arguments:
        ranks {list} -- list of ranks
        k {integer} -- k for top-k accuracy
    """
    top_k = 0
    mrr_sum = 0
    map_sum = 0
    total_reports = len(ranks)

    for rank in ranks:
        if rank <= k:
            top_k += 1
            # Calculate MRR
            mrr_sum += 1.0 / rank
            # Calculate MAP
            map_sum += 1.0 / rank

    accuracy = top_k / total_reports
    mrr = mrr_sum / total_reports
    map_score = map_sum / total_reports

    return accuracy, mrr, map_score


class CodeTimer:
    """ Keeps time from the initalization, and print the elapsed time at the end.

        Example:

        with CodeTimer("Message"):
            foo()
    """

    def __init__(self, message=""):
        self.message = message

    def __enter__(self):
        print(self.message)
        self.start = timeit.default_timer()

    def __exit__(self, exc_type, exc_value, traceback):
        self.took = timeit.default_timer() - self.start
        print("Finished in {0:0.5f} secs.".format(self.took))

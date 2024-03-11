# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 18:29:41 2020

@author: Asus
"""
from collections import Counter

import numpy as np

import sys
from gensim.models import word2vec
import csv
import nltk.corpus
import string
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
import nltk
import pickle as pickle
from collections import OrderedDict
from nltk.corpus import wordnet as wn
import networkx as nx
import matplotlib.pyplot as plt

stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(string.punctuation)
stopwords.append('')

node=[]
vertex1=[]
vertex2=[]



#Algorithm performance
def calculate_algorithm_performance(true_positives, true_negatives, false_positives, false_negatives):
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    fmeasure = 2 * ((precision * recall) / (precision + recall))
    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives+ false_negatives)

    return precision, recall, fmeasure, accuracy

def check_groundtruth_homographs(preprocessed_sentence):
    'Import the data...'
    file_homograph = "crowd_homographs_50.pickle"
    homographs = pickle.load(open(file_homograph,'rb'))
    found_homographs = []
    for word in preprocessed_sentence:
        if word in homographs:
            found_homographs.append(word)
    return homographs

#paper's implementation
def measure_overlap_bows1(a, b, threshold): 
    """Check if a and b are matches."""
    tokens_a = [strip_punctuation(token) for token in nltk.word_tokenize(a) if strip_punctuation(token) not in stopwords]
    tokens_b = [strip_punctuation(token) for token in nltk.word_tokenize(b) if strip_punctuation(token) not in stopwords]
    
    # Calculate Jaccard similarity
    ratio = len(set(tokens_a).intersection(tokens_b)) / float(len(set(tokens_a).union(tokens_b)))
    return (ratio >= threshold)

#part of the paper's implementation
def filter_similar_definitions(definition_list):
    remove_items = []
    for i in range(len(definition_list)+1):
        for definition in range(i+1,len(definition_list)):
            # Remove definitions with high path similarity
            a=definition_list[definition][0]
            b=definition_list[i][0]
            similarity = b.path_similarity(a)
            #if similarity is not None and similarity >= 0.30:
                
            if similarity is not None and similarity >= 0.10:
            #if similarity >= 0.30:
                remove_items.append(definition_list[definition][0])
            
            # Remove items with word overlap > 10% in definitions after stopwordremoval
            if measure_overlap_bows1(definition_list[i][1], definition_list[definition][1], 0.1) == True:
                remove_items.append(definition_list[definition][0])
    remove_items = list(OrderedDict.fromkeys(remove_items))
    distinct_defs = []
    for item in definition_list:
        if item[0] not in remove_items:
            distinct_defs.append(item)
    return distinct_defs


#Method implemented by the paper
def check_homography(word):
    definitions = []
    if word not in stopwords:
        definitions = [[n, n.definition()] for n in wn.synsets(word)]
        definitions = filter_similar_definitions(definitions)
        if len(definitions) >= 2:
            return True
        
def check_homography1(word,pos_tags):

    definitions = []
    d=[]
    if word not in stopwords:
        for n in wn.synsets(word):
            syn1=n.name()
            syn=n
            syn_list=syn1.split('.')
            if syn_list[1]==pos_tags:
                d.append(syn)
                d.append(n.definition())
            if len(d)>0:
                definitions.append(d)
                d=[]
        definitions = filter_similar_definitions(definitions)
        if len(definitions) >= 2:
            return True
    '''
    if word not in stopwords:
        for n in wn.synsets(word):
            if n.lemmas()[0].name() == word:
                syn1=n.name()
                syn=n
                #syn=wn.synsets(n)
                syn_list=syn1.split('.')
                if syn_list[1]==pos_tags:
                    d.append(syn)
                    d.append(n.definition())
            if len(d)>0:
                definitions.append(d)
                d=[]
        definitions = filter_similar_definitions(definitions)
        if len(definitions) >= 2:
            return True
    '''
def strip_punctuation(token):
    new_token = token.lower()
    for i in string.punctuation:
        if i in new_token:
            new_token = token.strip(' ')
            new_token = new_token.split(i)
            new_token = ' '.join(new_token)
    return new_token

def retrieve_synsets(word,pos_tag):
    sense=[]
    definitions=[]
    '''
    for n in wn.synsets(word):
        if n.lemmas()[0].name() == word:
            syn=n.name()
            syn_list=syn.split('.')
            if syn_list[1]==pos_tag:
                sense.append(syn)
                definitions.append(n.definition())
    return sense, definitions
    '''
    
    for n in wn.synsets(word):
        syn=n.name()
        syn_list=syn.split('.')
        if syn_list[1]==pos_tag:
            sense.append(syn)
            definitions.append(n.definition())
    return sense, definitions
    

def retrieve_synsets1(word):
    sense=[]
    definitions=[]
    
    for n in wn.synsets(word):
        sense.append(n.name())
        definitions.append(n.definition())
    return sense, definitions
    '''
    
    for n in wn.synsets(word):
        if n.lemmas()[0].name() == word:
            syn=n.name()
            sense.append(syn)
            definitions.append(n.definition())
    return sense, definitions
    '''
def measure_overlap_bows(tokens_a, tokens_b, threshold):
    # Calculate Jaccard similarity
    ratio = len(set(tokens_a).intersection(tokens_b)) / float(len(set(tokens_a).union(tokens_b)))
    return (ratio >= threshold)

def create_senseclusters():
    G = nx.Graph()
    for i in range(len(node)):
        G.add_node(node[i])
    for i in range(len(vertex1)):
        G.add_edge(vertex1[i],vertex2[i])

    #nx.draw(G, with_labels=True, node_color='orange', node_size = 100)
    nx.draw(G, with_labels=True, node_color='orange')
    plt.show()
    print("The number of connected component: "+str(nx.number_connected_components(G)))
    c=nx.number_connected_components(G)
    G.clear()
    return c

def intersection_bet_list(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def unique(list1): 
      
    # insert the list to the set 
    list_set = set(list1) 
    # convert the set to the list 
    unique_list = (list(list_set)) 
    return unique_list 


def checkSharedwords(tokens_a, tokens_b):
    count=0
    d1 = unique(tokens_a)
    d2 = unique(tokens_b)
    count=len(set(d1).intersection(d2))
    if count>0:
        return True
    return False

def checkSynonyms(tokens_a, tokens_b):
    count=0
    for a in tokens_a:
        for b in tokens_b:
            d1 = wn.synsets(a)
            d2 = wn.synsets(b)
            
            d1 = unique(d1)
            d2 = unique(d2)
            count=len(set(d1).intersection(d2))
            
            if count>0:
                break
        if count>0:
            return True
    return False

def checkHypernymy(tokens_a, tokens_b):
    count=0
    hyp1=[]
    hyp2=[]
    for a in tokens_a:
        for b in tokens_b:
            synsets_a = wn.synsets(a)
            synsets_b = wn.synsets(b)
            for i in range(len(synsets_a)):
                hyp=synsets_a[i].hypernyms()
                if(len(hyp)>0):
                    hyp1.append(hyp)
            for j in range(len(synsets_b)):
                hyp=synsets_b[j].hypernyms()
                if(len(hyp)>0):
                    hyp2.append(hyp)
            flat_list1 = [item for sublist in hyp1 for item in sublist]
            flat_list2 = [item for sublist in hyp2 for item in sublist]
            
            flat_list1=unique(flat_list1)
            flat_list2=unique(flat_list2)
            count=len(set(flat_list1).intersection(flat_list2))
            
            if count>0:
                hyp1=[]
                hyp2=[]
                break
        if count>0:
            return True
    return False



def method1(tokens_a,tokens_b,a,b):
    A= wn.synset(a)
    B= wn.synset(b)
    similarity = B.path_similarity(A)
    node.append(a)
    node.append(b)
    #if (similarity is not None and similarity >= 0.3) or (measure_overlap_bows(tokens_a,tokens_b, 0.1) == True):
    if (similarity is not None and similarity >= 0.1) or (measure_overlap_bows(tokens_a,tokens_b, 0.1) == True):
    #if (measure_overlap_bows(tokens_a,tokens_b, 0.1) == True):
        vertex1.append(a)
        vertex2.append(b)
    return

def method2(tokens_a,tokens_b,a,b):
    node.append(a)
    node.append(b)
    
    
    A= wn.synset(a)
    B= wn.synset(b)
    similarity = B.path_similarity(A)
    
    #if (similarity is not None and similarity >= 0.1) or checkSharedwords(tokens_a, tokens_b):
    #    vertex1.append(a)
    #    vertex2.append(b)
    
    #if checkSynonyms(tokens_a, tokens_b) and checkHypernymy(tokens_a, tokens_b):
    if checkSharedwords(tokens_a, tokens_b):
        vertex1.append(a)
        vertex2.append(b)
    elif checkSynonyms(tokens_a, tokens_b):
        vertex1.append(a)
        vertex2.append(b)
    #if checkHypernymy(tokens_a, tokens_b):
    #    vertex1.append(a)
    #    vertex2.append(b)
    return


def tokenizedefinitions(pos_tag, sense,definition_list,method_number):
    for i in range(len(definition_list)+1):
    #for i in range(len(definition_list)+1):
        for definition in range(i+1,len(definition_list)):
            a=definition_list[definition]
            b=definition_list[i]
            
            tokens_a = [strip_punctuation(token) for token in nltk.word_tokenize(a) if strip_punctuation(token) not in stopwords]
            tokens_b = [strip_punctuation(token) for token in nltk.word_tokenize(b) if strip_punctuation(token) not in stopwords]
            tokens_a=' '.join(tokens_a).split()
            tokens_b=' '.join(tokens_b).split()
            if method_number=="1":
                method1(tokens_a,tokens_b,sense[definition],sense[i])
            elif method_number=="2":
                method2(tokens_a,tokens_b,sense[definition],sense[i])
    c=create_senseclusters()
    if c>1:
        return True
    return False


def read_input(file):
    list1 = open(file, 'r').readlines()
    input_list = []
    for i in list1:
        input_list.append(i.strip("\n"))
    return input_list

def read_input1(file):
    list1 = open(file, 'r').readlines()
    input_list = []
    for i in list1:
        i=i.strip("\n")
        i = [k for j in i.split() for k in (j, ' ')][:-1] 
        i=i[0]
        i=i[:-2]
        input_list.append(i)
    return input_list

def retrieve_word_pos_tag(input1):
    print(input1)
    x = input1.split('#')
    word=x[0]
    pos_tag=x[1]
    return word,pos_tag


#TAKEN FROM PAPER'S IMPLEMENTATION

def calculate_sentence_performance(correct_words,found_words,all_words):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for word in found_words:
        if word in correct_words:
            tp += 1
        else:
            fp += 1
    for word in all_words:
        if word not in found_words:
            if word not in correct_words:
                tn += 1
            else:
                fn += 1
    return [tp,tn,fp,fn]


def preprocess(text):
    result = [strip_punctuation(token) for token in nltk.word_tokenize(text)\
                if strip_punctuation(token) not in stopwords]
    return result

def identify_homographs2(words, sentence):
    homographs=[]
    for i in range(len(words)):
        sense, definitions=retrieve_synsets1(words[i])
        #sense, definitions=retrieve_synsets(words)
        pos_tag=""
        if (tokenizedefinitions(pos_tag, sense, definitions,"1")):
            homographs.append(words[i])
            node.clear()
            vertex1.clear()
            vertex2.clear()
    return homographs

def identify_homographs1(words, sentence):
    homographs = [w for w in words if check_homography(w) == True]
    return homographs

def find_homographs2(preprocessed_sentence, full_sentence):
    homographs = [h for h in identify_homographs2(preprocessed_sentence, full_sentence) if h.isdigit() != True]
    return homographs

def find_homographs1(preprocessed_sentence, full_sentence):
    homographs = [h for h in identify_homographs1(preprocessed_sentence, full_sentence) if h.isdigit() != True]
    return homographs


def main():
    
    file1="bln-all.txt"
    file2="dev-hom.txt"
    file3="dev-pol.txt"
    
    file4="bln-hom.txt"
    file5="bln-pol.txt"
    input_list=read_input(file1)
    dev_homo_list=read_input1(file2)
    dev_pol_list=read_input1(file3)
    
    bln_homo_list=read_input1(file4)
    bln_pol_list=read_input1(file5)
    
    #devset_words=[]
    #for i in range(len(dev_homo_list)):
    #    devset_words.append(dev_homo_list[i])
    #for i in range(len(dev_pol_list)):
    #    devset_words.append(dev_pol_list[i])
    
    blnset_words=[]
    for i in range(len(bln_homo_list)):
        blnset_words.append(bln_homo_list[i])
    for i in range(len(bln_pol_list)):
        blnset_words.append(bln_pol_list[i])
    
    
    print("Enter the method number to cluster senses:")
    method_number = input()
    homo=[]
    homo_paper=[]
    poly_paper=[]
    
    total_homonym1 = [0.00000001,0.00000001,0.00000001,0.00000001]
    for i in range(len(blnset_words)):
        word,pos_tag=retrieve_word_pos_tag(blnset_words[i])
        sense, definitions=retrieve_synsets(word,pos_tag)
        if(tokenizedefinitions(pos_tag, sense, definitions,method_number)):
            homo.append(blnset_words[i])
        node.clear()
        vertex1.clear()
        vertex2.clear()
        
        #base paper's method
        #if(check_homography(word)):
        if(check_homography1(word,pos_tag)):
            #homo_paper.append(devset_words[i])
            homo_paper.append(blnset_words[i])
    
    
    #Performance measure
    
    result_1=calculate_sentence_performance(bln_homo_list,homo,blnset_words)
    #result_1=calculate_sentence_performance(bln_homo_list,homo_paper,blnset_words)
    #result_1=calculate_sentence_performance(dev_homo_list,homo,devset_words)
    #result_1=calculate_sentence_performance(dev_homo_list,homo_paper,devset_words)
    print(result_1)
    #sent_result_2 = calculate_sentence_performance(groundtruth_sentence, homographs2_results, prepped_sent)
    
    
    for i in range(len(result_1)):
        #total_homographs1[i] += sent_result_1[i]
        total_homonym1[i] += result_1[i]
    
    
    perf_homographs_1 = calculate_algorithm_performance(total_homonym1[0], total_homonym1[1], total_homonym1[2], total_homonym1[3])
    print("Precision alg_1: %.3f \t Recall: %.3f \t F-measure: %.3f \t Accuracy: %.3f"%(perf_homographs_1[0], perf_homographs_1[1], perf_homographs_1[2], perf_homographs_1[3]))
    
    
    
    
    #Error Analysis:
    print("Error Analysis:")
    detected_homo=intersection_bet_list(bln_homo_list, homo_paper)
    print(detected_homo)
    not_detected_homo=set(bln_homo_list) - set(homo_paper)
    print(not_detected_homo)
    false_detected_homo= set(homo_paper)-set(bln_homo_list)
    print(false_detected_homo)
    
    
    # BA18 method and their implementation. It is kept as commented code to separate it from the main system
    '''
    # Prevents 0 multiplication error
    total_homographs1 = [0.00000001,0.00000001,0.00000001,0.00000001]
    total_homographs2 = [0.00000001,0.00000001,0.00000001,0.00000001]

    
    with open('annotated_homographs_english_v4.csv',encoding="utf8") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        reduced_rows = []
        unique_sentences = []
        unique_ids = []
        duplicate_ids = []
        replace_list = []
        for row in spamreader:
            #print(', '.join(row))
            new_row = [row[-2], row[7], row[13], row[0]]
            reduced_rows.append(new_row)
            if new_row[0] not in unique_sentences:
                unique_sentences.append(new_row[0])
                if new_row[-1] not in unique_ids:
                    unique_ids.append(new_row[-1])
            else:
                if new_row[-1] not in unique_ids and new_row[-1] not in duplicate_ids:
                    duplicate_ids.append(new_row[-1])
                    replace_list.append([new_row[-1],unique_ids[unique_sentences.index(new_row[0])]])

    
    for sentence in unique_sentences:
        prepped_sent = preprocess(sentence)
        # Create list of all homographs for each method
        groundtruth_sentence = check_groundtruth_homographs(prepped_sent)
        
        #homographs1_results = find_homographs1(prepped_sent, sentence)
        
        homographs2_results = find_homographs2(prepped_sent, sentence)
        # Calculate true positives, true negatives, false positives and false negatives
        #sent_result_1 = calculate_sentence_performance(groundtruth_sentence, homographs1_results, prepped_sent)
        sent_result_2 = calculate_sentence_performance(groundtruth_sentence, homographs2_results, prepped_sent)
    
    
        for i in range(len(sent_result_2)):
            #total_homographs1[i] += sent_result_1[i]
            total_homographs2[i] += sent_result_2[i]
    
    #perf_homographs_1 = calculate_algorithm_performance(total_homographs1[0], total_homographs1[1], total_homographs1[2], total_homographs1[3])
    #print ("Precision alg_1: %.3f \t Recall: %.3f \t F-measure: %.3f \t Accuracy: %.3f"%(perf_homographs_1[0], perf_homographs_1[1], perf_homographs_1[2], perf_homographs_1[3]))
    
    perf_homographs_2 = calculate_algorithm_performance(total_homographs2[0], total_homographs2[1], total_homographs2[2], total_homographs2[3])
    print ("Precision alg_2: %.3f \t Recall: %.3f \t F-measure: %.3f \t Accuracy: %.3f"%(perf_homographs_2[0], perf_homographs_2[1], perf_homographs_2[2], perf_homographs_2[3]))
    '''
if __name__ == "__main__":
    main()
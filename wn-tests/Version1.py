# !/usr/local/bin/python

# if it is a tangoble abstract object, if it has feature tangible - it is capbably of being an abstract thing as physical object
# if it is tangible and a container it is a set of things which support set operations.
print("for a NP find noun's ontology features, find modifiers scaling directly \n",
      " and/or find physical properties from wnet either by traveling up hypernyms \n",
      " and testing lemmas against trips\n or directly from wn hypernyms\n find intensifiers \n" )

print("Outline: what are common metaphoric gesturs Cienki and Lhommet, ML ? \n",
      "and database but with meanngful use of gesture, coverage and creative control")

#from PySide import QtGui
# from CII import CII as CII_Obj

from pytrips.nodegraph import type_to_dot
from pytrips.structures import TripsRestriction, TripsType, TripsSem
from pytrips.ontology import load as tload
ont = tload()
import re
from nltk.corpus import wordnet as wn
# print(wn.synset('group.n.01'))
import spacy

# from textblob import TextBlob as BL

from pytrips.ontology import load as tload
ont = tload()

from pytrips.helpers import helpers as hlp
spacyTable=hlp.make_spacy_pos_table()


# from spacy.symbols import nsubj, VERB, ADV, ADJ, NOUN

from spacy.lemmatizer import Lemmatizer

from spacy import displacy

nlp = spacy.load("en_core_web_lg")

from nltk.wsd import lesk as lk
from nltk.corpus import verbnet as vn

hypo = lambda s: s.hyponyms()
hyper = lambda s: s.hypernyms()

advDict = {}

#store adverb dictionary
with open("Adv.txt", 'r') as f:
    for line in f:
        items = line.split()
        advDict[items[0]] = items[1]

# extract the noun phrases in a sentence
def processNP(chunk):
    root = chunk.root
    # print(chunk.text.split())
    mods = chunk.text
    wrd = chunk.root.text
    mods = chunk.text.replace(wrd, '').split()
    return (root,mods,text)





def collectSimilarTo(wrd,pos='a'):
    bigSet = set([])
    for syn in wn.synsets(wrd, pos): # create a list of words that are similar (synonyms) for the adjective
        for l in syn.lemmas():
            bigSet.add(l.name()) # using a list but really should use a set to simplify membership check below
        for syn in syn.similar_tos():
            for l in syn.lemmas():
                bigSet.add(l.name())
    return bigSet


def collectFeaturesFromLemmas(wlst,pos):
    # sSet = set(ont['w::' + adj,'a'])
    # sSet=set([])
    sSet= set([y for wrd in wlst for y in ont['w::' + wrd,pos]])
    # sSet = sSet.union(set(ont['w::' + wrd,pos]) for wrd in wlst)
    return collectFeatures(sSet)


ILLEGAL_SEMF = ['?', '-', 'O']


def collectFeatures(oSet):
    fSet = set([])
    for o in oSet:
        for f in o.sem.sem.keys():
            if (o.sem.sem[f] == '-'):
                continue
            else:
                if (o.sem.sem[f] == '+'):
                    fSet.add(f)
                else:
                    semF = o.sem.sem[f]
                    if isinstance(semF, list):
                        # TODO: CARO, ASK STACY ABOUT THIS.
                        # CURRENTLY ADDS MULTIPLE OPTIONS IF MULTIPLE ARE GIVEN.
                        for el in semF:
                            if el not in ILLEGAL_SEMF:
                                fSet.add(el)
                        continue
                    else:
                        fSet.add(o.sem.sem[f])
    return fSet


def WordToHypernymLemmas(word, pos):
    setHypernyms = set(wn.synsets(word,pos))
    setLemmas = set([])
    for syn in setHypernyms:
        setHypernyms = setHypernyms.union(set(syn.closure(hyper, depth=2)))
    for syn in setHypernyms:
        setLemmas = setLemmas.union(set(syn.lemma_names()))
    return setLemmas, setHypernyms


def selectGestureFromOntology(s):
    return s


def selectGestureFromLemmas(s):
    return s



def process_sent(row):
    sent = row.transcript
    spacyDoc = nlp(sent)  # prcess sentence using spacy
    nchunks = spacyDoc.noun_chunks
    o_featurelst =[]
    w_featurelst = []
    gestDict = {'utterance': sent}
    npInc = 1
    # NPlst = extractNPs(nchunks) # extract all the Noun Phrase
    for chunk in nchunks:
        gestDict = process_NP(chunk, npInc, spacyDoc,gestDict)
        npInc += 1
    return gestDict


def process_NP(chunk, npInc, spacyDoc, gestDict):
    root = chunk.root
    # print(chunk.text.split())
    mods = chunk.text
    wrd = chunk.root.text
    mods = chunk.text.replace(wrd, '').split()
    root = chunk.root

    # lesk???
    gFeats = set([])
    gestDict[npInc] = {'text': chunk.text, 'root': root.text, 'mods': mods}
    otypes = ont['w::' + root.text, 'n']
    exp_otypes = set(otypes)
    for ext in otypes:
        exp_otypes.add(ext.parent)
    wnLemmas, wnNyms = WordToHypernymLemmas(root.text, 'n')
    exp_otypes = set(y for w in wnLemmas for y in ont['w::' + w, 'n'])
    # print('otypes: ', otypes, type(otypes))
    if len(otypes) > 0:              #check
        o_featurelst = collectFeatures(otypes)
        gFeats = selectGestureFromOntology(o_featurelst)
    else:
        if len(exp_otypes) > 0:
            o_featurelst = collectFeatures(exp_otypes)
            gFeats = selectGestureFromOntology(o_featurelst)
        else:
            gFeats = selectGestureFromLemmas(wnLemmas)

    for child in root.children:
        if child.dep_ == 'amod':
            # modSet = modSet.union(collectFeatInfo(collectSimilarToAdjs(child.text)))
            gestDict[npInc][child.text] = collectFeaturesFromLemmas(collectSimilarTo(child.text, 'a'), 'a')
        if child.dep_ == 'advmod':
            lemmas = set([l for x in wn.synsets(child.text, 'r') for l in x.lemma_names()])
            gestDict[npInc][child.text] = collectFeaturesFromLemmas(lemmas, 'r')
        if child.dep_ == 'prep':
            prepOnts = set(ont['w::' + child.text])
            gestDict[npInc][child.text] = collectFeatures(prepOnts)
        if child.dep_ == 'acl':
            lemmas = set([l for x in wn.synsets(child.text, 'v') for l in x.lemma_names()])
            gestDict[npInc][child.text] = collectFeaturesFromLemmas(lemmas, 'v')
        if child.dep_ == 'relcl':
            lemmas = set([l for x in wn.synsets(child.text, 'v') for l in x.lemma_names()])
            gestDict[npInc][child.text] = collectFeaturesFromLemmas(lemmas, 'v')
    return gestDict


def proc_by_verb(doc):
    verbs = set()
    for possible_subject in doc:
        if possible_subject.dep == nsubj and possible_subject.head.pos == VERB:
            verbs.add(possible_subject.head)
    print("verbs: ", verbs)





                    # feats = list(otype.sem.sem.keys())
# features =        wrd = chunk.root.text
# mods = chunk.text.replace(wrd,'').split()

filename = input("\nFilename: ")

try:
    tfPtr = open(filename, 'r')
except:
    tfPtr = open('tests.txt','r')

import pprint

pp = pprint.PrettyPrinter(indent=4, compact=True)

for sent in tfPtr:
    pp.pprint(process_sent(sent))


# doc = nlp('There is an audience of Republicans, Independents and Democrats')
# doc = nlp('The manufacturers of American cars and Japanese cars and German cars are fast')

def procAssertions(doc):
    winc = 0
    for token in doc:
        if token.dep_ == 'ROOT':
            # print("({} {} {})".format(token.tag_, winc - token.n_lefts, winc + token.n_rights))
            for tok in token.subtree:
                print("({} {} {} {})". format('word',tok.text, winc, winc+1))
                # print(tok.tag_,' ',tok.dep_)
                sub = tok.subtree
                seq= [s for s in sub]
                # print(tok.text, [child for child in tok.children], tok.n_lefts, tok.n_rights)
                # print("({} {} {} {} {})".format(tok.dep_, tok.tag_, seq, winc - tok.n_lefts, winc + tok.n_rights + 1))
                print("({} {} {} {} {})".format(tok.dep_, tok.tag_, seq, tok.left_edge.i, tok.right_edge.i + 1))
                winc += 1


# procAssertions(doc)

#close(tfPtr))


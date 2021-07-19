# if it is a tangoble abstract object, if it has feature tangible - it is capbably of being an abstract thing as physical object
# if it is tangible and a container it is a set of things which support set operations.
# print("for a NP find noun's ontology features, find modifiers scaling directly \n",
#      " and/or find physical properties from wnet either by traveling up hypernyms \n",
#      " and testing lemmas against trips\n or directly from wn hypernyms\n find intensifiers \n" )

# print("Outline: what are common metaphoric gesturs Cienki and Lhommet, ML ? \n",
#      "and database but with meanngful use of gesture, coverage and creative control")
DEBUG = False


def debprint(*args):
    if DEBUG:
        print(*args)


import os
import logging
import logging.config
from collections import defaultdict as ddict

from datetime import datetime

# from utils import utils_parse, utils_wordnet, utils_xml, utils_gen, utils_acoustic
# from config import params
FLIST = {}
TAGS = set([])
DEPS = set([])
SCONJ = {
    "than": "comparison",
    "rather than": "comparison",
    "whether": "comparison",
    "as much as": "comparison",
    "whereas": "comparison",
    "that": "relative",
    "whatever": "relative",
    "which": "relative",
    "whichever": "relative",
    "after": "time",
    "aa soon as": "time",
    "as long as": "time",
    "before": "time",
    "by the time": "time",
    "now that": "time",
    "once": "time",
    "since": "time",
    "till": "time",
    "until": "time",
    "when": "time",
    "whenever": "time",
    "while": "time",
    "though": "concession",
    "although": "concession",
    "even though": "concession",
    "who": "relativepn",
    "whoever": "relativepn",
    "whom": "relativepn",
    "whomever": "relativepn",
    "whose": "relativepn",
    "where": "place",
    "wherever": "place",
    "if": "condition",
    "only if": "condition",
    "unless": "condition",
    "provided that": "condition",
    "assuming that": "condition",
    "in case": "condition",
    "in case that": "condition",
    "lest": "condition",
    "how": "manner",
    "as though": "manner",
    "as if": "manner",
    "because": "reason",
    "since": "reason",
    "so that": "reason",
    "in order": "reason",
    "in order that": "reason",
    "that": "reason",
    "as": "reason",
}

Gsent = ""

# Import Pyke, import engines . all engines are all globals
# from pyke import knowledge_engine
# from engine import rules
#
# from engine.data import InitSemantics as sem
# nvbg_engine = knowledge_engine.engine(rules)


# execfile(os.path.join(dataPath, 'InitDefaultBehaviorBMLMapping.py'))

# setup the logger
now = datetime.now().strftime("%y%m%d_%H%M%S")
# logging.config.fileConfig('config/log.conf', defaults={'logfilename': './tmp/' + now + '.log'})
# logger = logging.getLogger("Cerebella_II")

# from PySide import QtGui
# from CII import CII as CII_Obj

from pytrips.nodegraph import type_to_dot
from pytrips.structures import TripsRestriction, TripsType, TripsSem
from pytrips.ontology import load as tload
from pytrips.helpers.helpers import make_spacy_pos_table, get_wn_key, ss_to_sk, all_hypernyms, all_hyponyms

ont = tload(skip_lexicon=False, use_gloss=True)

tagConvert = make_spacy_pos_table()
debprint(make_spacy_pos_table())


def convert_wn_to_trips(pos):
    if pos == 'a': return 'adj'
    if pos == 'r': return 'adv'
    return pos


import re

# import xml.etree.ElementTree as ET
OntWords = ddict(list)
OntPhrases = ddict(list)
from xml.etree import cElementTree as ET

otree = ET.parse('TRIPS-ontology.xml')
for typ in otree.iter('ONTTYPE'):
    t_ont = typ.attrib['name']
    wntype = None
    for w in typ.iter('Mapping'):
        if w.attrib.get('to') == 'wordnet':
            wntype = w.attrib.get('name', '')
    for w in typ.iter('WORD'):
        OntWords[w.attrib['name']] = (t_ont, wntype)
        if '_' in w.attrib['name']:
            OntPhrases[w.attrib['name']] = (t_ont, wntype)

from nltk.corpus import wordnet as wn

import spacy

nlp = spacy.load("en_core_web_lg")

from pytrips.helpers import helpers as hlp

spacyTable = hlp.make_spacy_pos_table()

# from spacy.symbols import nsubj, VERB, ADV, ADJ, NOUN

# from spacy.lemmatizer import Lemmatizer

from spacy import displacy

from nltk.wsd import lesk as lk
from nltk.corpus import verbnet as vn

hyper = lambda s: s.hypernyms()

advDict = {}

NPtokens = set([])


# print(ont['w::dog','n'])
# print(ont['w::very','adv'])
# print(ont['w::important','adj'])
# print(ont['w::run','v'])


def dictprint(d):
    for key in d.keys():
        print(key, ":", d[key])


class CII:
    """ main class """

    def __init__(self, debug_print=False):
        """ Initialization """
        nlp = spacy.load("en_core_web_lg")

        self.debug_print = debug_print
        hyper = lambda s: s.hypernyms()
        hypo = lambda s: s.hyponyms()

        advDict = {}
        # store adverb dictionary
        # with open("Adv.txt", 'r') as f:
        #     for line in f:
        #         items = line.split()
        #         advDict[items[0]] = items[1]

        # sem.loadSemanticsFacts(self.getEngine())
        # logger.info("Initialization done.")

    def getEngine(self):
        """ return the Pyke inference engine used by this instance of Cerebella """
        return nvbg_engine

    def resetEngine(self):
        """ resets the content of the Pyke inference engine used by this instance of Cerebella """
        self.getEngine().reset()

    def jsontrips(self):
        import jsontrips
        wrds = jsontrips.words()
        entries = wrds['entries']
        self.wdict = {}
        for entry in entries.keys():
            if '<' in entry:
                continue
            wrd = entry.lower().rstrip('1234567890')
            self.wdict[wrd] = {'pos': entries[entry]['pos']}
            for sens in entries[entry]['senses']:
                self.wdict[wrd]['parent'] = []
                try:
                    self.wdict[wrd]['parent'].append(sens['lf_parent'])
                except:
                    continue

    def collectSimilarTo(self, wrd, pos='a'):
        bigSet = set([wrd, ])
        for syn in wn.synsets(wrd, pos):  # create a list of words that are similar (synonyms) for the adjective
            for l in syn.lemmas():
                bigSet.add(l.name())  #
                for syn in syn.similar_tos():
                    for l in syn.lemmas():
                        bigSet.add(l.name())
        # print('bigSet',wrd, bigSet)
        return bigSet

    def collectFeaturesFromLemmas(self, wlst, ont_pos):
        sSet = set([y for wrd in wlst for y in ont['w::' + wrd, ont_pos]])
        return self.collectFeatures(sSet)

    def collectFeatures(self, oSet):  # these are the semantic features from trips - dbl check
        fSet = set([])
        for o in oSet:
            if o == None:
                continue
            for f in o.sem.sem.keys():
                # print(o, f, o.sem.sem[f])
                if (o.sem.sem[f] == '-'):
                    continue
                elif (o.sem.sem[f] == '+'):
                    fSet.add(f)
                else:
                    # print(f,o.sem.sem[f],type(o.sem.sem[f]))
                    fSet.add(f)
                    if isinstance(o.sem.sem[f], str):
                        fSet.add(o.sem.sem[f])
                        return fSet
                    else:
                        return fSet.union(set(o.sem.sem[f]))
        # print("fSet", fSet)
        return fSet

    def WordToHypernymLemmas(self, word, pos):
        isetHypernyms = set(wn.synsets(word, pos))
        setHypernyms = set(wn.synsets(word, pos))

        # print(isetHypernyms)
        setLemmas = set([])
        for syn in isetHypernyms:
            setHypernyms = setHypernyms.union(set(syn.closure(hyper, depth=2)))
            for syn in setHypernyms:
                setLemmas = setLemmas.union(set(syn.lemma_names()))
        return setLemmas, setHypernyms

    def selectGestureFromOntology(self, s):
        return s

    def selectGestureFromLemmas(self, s):
        return s

    # Go through the parse
    # Collect features of each token along the way.
    def proc_parse(self, doc):
        winc = 0
        iseq = []
        self.SEtok = {}
        # this for loop just asserts each word - not needed
        for token in doc:
            self.SEtok[token] = (winc, winc + 1)
            winc = winc + 1

        token_asstns = set([])
        totFeats = set()
        tree_ont = {}
        for token in doc:
            # print(">>>>>>>>>>>>>>>>>>>>>>>>>> TOKEN: ", token)
            tokenFeats = set()
            if token.dep_ == 'ROOT':
                iseq = [s.text for s in token.subtree]
                # process by phrases
                # i think this is because some ontology entries are pharses connected by underlines
                phrase_feats = self.proc_phrase(iseq)  # TODO this currently does nothing
                # process by noun phrases
                np_feats = self.proc_NPs(iseq, doc)
                tokenFeats = tokenFeats.union(phrase_feats, np_feats)
                tree_ont[token.text] = tokenFeats
                for tok in token.subtree:
                    # print(">>>>>>>>>>>>> subtree token: ", tok)
                    # print(tok.tag_,' ',tok.dep_)
                    sub = tok.subtree
                    subseq = [s.text for s in sub]
                    # this just asserts the synatctic structure
                    # syn_assert = "({} {} {} {} {})".format(tok.tag_, tok.dep_, subseq, tok.left_edge.i, tok.right_edge.i + 1)
                    # not using pyke so this just prints
                    # process the a suntactic structure - i would this would duplicate analysis of noun phrases
                    subtok_feats = self.proc_other(tok, tok.left_edge.i, tok.right_edge.i + 1)
                    # if tok.tag_ in ['NN', 'NNS', 'NNP', 'NNPS']:
                    #     print('Noun: ',tok,[x for x in tok.children])
                    totFeats = totFeats.union(tokenFeats)
                    tree_ont[tok] = subtok_feats

        # if len(totFeats) > 0:
        if tree_ont:
            return tree_ont
        else:
            # print("NO  ROOT ", doc.text)
            return {}

    def proc_phrase(self, seq, inc=0, phrase_feats=set()):
        # print(">>>>>>>>> PROC PHRASE:", seq)
        # this is supposed to scan the original seq of words looking for the ontological phrases
        phrase_feats = set()
        for j in range(len(seq) - 2):
            start = j + inc
            for i in [3, 2]:
                phr = '_'.join(seq[start:start + i])
                ophr = OntPhrases[phr]
                # TODO: this currently always returns an empty list?
                if ophr:
                    o = ophr[0]
                    if ont['ont::' + o] != None:
                        feats = self.collectFeatures(set([ont['ont::' + o]]))  # get ontological features
                        # just print
                        # self.pyke_assert("({},{} {} {} {} {})".format('function', 'Phrase', ' '.join(seq[start:start+i]), feats, start, start+i),feats)
                        phrase_feats = phrase_feats.union(feats)
                    # print(start+i,seq)
                    self.proc_phrase(seq, start + i, phrase_feats)
                    continue
                else:
                    # print("OPHR empty for phrase: ", seq)
                    continue
        # print(">>>>>>>>>>>>> PHRASE FEATS: ", phrase_feats)
        return phrase_feats

    def proc_NPs(self, seq, doc):
        # print("PROC NPS: ", seq)
        # print("Entering NPS")
        # doc = nlp(sent)  # prcess sentence using spacy
        nchunks = doc.noun_chunks
        start = 0
        end = 0
        # NPlst = extractNPs(nchunks) # extract all the Noun Phrase
        feats = set([])
        for chunk in nchunks:
            # print(">>>>>>>>>>> CHUNK: ", chunk)
            b = list(chunk.text.split())
            sub = [(i, i + len(b)) for i in range(end, len(seq)) if seq[i:i + len(b)] == b]
            if len(sub) < 1:
                return set(sub)
            else:
                start = sub[0][0]
                end = sub[0][1]
                # just process indiviual noun phrases
                feats = feats.union(self.proc_NP(chunk, seq, start, end + 1))
                conjlst = []
                # look for conjunctions since they have special rhetorical and metaphoric meaning
                if chunk.root.dep_ in ['nsubj', 'pobj', 'dobj']:  # and len(chunk.root.children) > 0:
                    if 'conj' in [child.dep_ for child in chunk.root.children]:
                        conjlst = [(chunk.root, (start, end)), ]
                        conjlst.extend(self.find_conj(chunk.root.children))
                        toklst = [t[0].text for t in conjlst]
                        starts = [t[1][0] for t in conjlst]
                        ends = [t[1][1] for t in conjlst]
                        syn_assert = "({} {} {} {} {} {})".format('function', 'noun_sequence', chunk.root.dep_, toklst,
                                                                  starts[0], ends[-1])
                        # print("WHAT IS SYN ASSERT????? ", syn_assert)
                        # feats = feats.union(self.pyke_assert(syn_assert))
        # print("NP RETURNING: ", feats)
        return feats

    def proc_NP(self, chunk, seq, start, end):
        # print(">>>>>entering NP", chunk.text)
        if chunk.root.pos_ == 'PRON':
            # print("returning empty for pron")
            return set()
        if chunk.root.pos_ == 'PROPN':
            # print("returning empty for propn")
            return set()

        root = chunk.root
        wrd = chunk.root.text
        text = chunk.text
        # lesk???
        #            SOMETHING WEIRD HERE - featTots not used###############
        featTots = set(self.proc_ont(wrd, start, end, pos='n'))  # ont features of root word
        end = start
        for child in chunk:
            # print("CHILD: ", child)
            NPtokens.add(child)  # hmmm?
            feats = []
            dep = child.dep_
            # just find start and end of words
            b = child.text if isinstance(child.text, (list, tuple)) else [child.text]
            sub2 = [(i, i + len(b)) for i in range(end, len(seq)) if seq[i:i + len(b)] == b]
            if len(sub2) > 0:
                start = sub2[0][0]
            end = sub2[0][1]
            # eed to treat adjectival modifiers uniquely in wordnet
            if dep == 'amod':
                similars = self.collectSimilarTo(child.text, 'a')  # adj synonyms in wordnet
                feats = self.collectFeaturesFromLemmas(similars, 'adj')  # get ont features fro all the synoyms
            # need to treat adverbial modifiers differently in wordnet
            elif dep == 'advmod':
                lemmas = set([l for x in wn.synsets(child.text, 'r') for l in x.lemma_names()])
                feats = self.collectFeaturesFromLemmas(lemmas, 'adv')

            if len(feats) > 0:
                self.pyke_assert("({} {} {} {} {} {} {})".format('function', 'Ont', child.text, dep, feats, start, end),
                                 feats)
                featTots = featTots.union(set(feats))
        # print("NP FEATURE TOTALS: ", chunk.text, featTots)
        return featTots

    def find_conj(self, children):
        # print("Entering CONJ")
        conj = None
        for child in children:
            if child.dep_ == 'conj':  # relying on spacy
                conj = child
        if conj == None:
            # print(1,[])
            return []
        lst = [c for c in conj.children]
        # print(conj, lst,[(conj,self.SEtok[conj]),])
        if lst == []:
            # print(2,lst,[(conj,self.SEtok[conj]),])
            return [(conj, self.SEtok[conj]), ]
        else:
            res0 = self.find_conj(lst)
            res1 = [(conj, self.SEtok[conj]), ]
            res1.extend(res0)
            # print(3,res1)
            return res1

    def proc_sc_phrase(self, token, start, end):  # but is spacy good enough to detect SC
        lphr = [child.text for child in token.lefts]
        lInc = len(lphr)
        rphr = [child.text for child in token.rights]
        rInc = len(rphr)
        lphr.append(token.text)
        lphr.extend(rphr)
        phr = " ".join(lphr)
        # print(phr)
        feats = SCONJ.get(phr)
        if feats != None:
            # print("SC FEATS?", feats)
            return feats
        return set()

    def proc_other(self, tok, start, end):
        # print("Entering OTHER", tok.text)
        totFeats = set()
        try:
            pos = tagConvert[tok.tag_]
            totFeats = self.proc_ont(tok.text, start, end, pos=pos)
        except:
            # print('Tag not found: ', tok.pos_)
            TAGS.add(tok.pos_)
        for child in tok.children:
            if child in NPtokens:  # don't duplicate effort on NPs
                continue
            feats = []
            dep = child.dep_
            pos = child.pos_
            DEPS.add(dep)
            start = child.left_edge.i
            end = child.right_edge.i + 1
            if dep == 'amod':
                feats = self.collectFeaturesFromLemmas(self.collectSimilarTo(child.text, 'a'), 'adj')
            elif dep == 'advmod':
                lemmas = set([l for x in wn.synsets(child.text, 'r') for l in x.lemma_names()])
                feats = self.collectFeaturesFromLemmas(lemmas, 'adv')
            elif dep == 'prep':
                prepOnts = set(ont['w::' + child.text])
                feats = self.collectFeatures(prepOnts)
            elif dep == 'acl':
                lemmas = set([l for x in wn.synsets(child.text, 'v') for l in x.lemma_names()])
                feats = self.collectFeaturesFromLemmas(lemmas, 'v')
            elif dep == 'relcl':
                lemmas = set([l for x in wn.synsets(child.text, 'v') for l in x.lemma_names()])
                feats = self.collectFeaturesFromLemmas(lemmas, 'v')
            elif dep == 'advcl':
                lemmas = set([l for x in wn.synsets(child.text, 'v') for l in x.lemma_names()])
                feats = self.collectFeaturesFromLemmas(lemmas, 'v')
            elif dep == 'cc':
                feats, start, end = self.proc_cc_phrase(child, start, end)
            elif pos == 'SCONJ':
                # print('>>>>>>>>>>>>>>>>>>>>>>>> SCONJ: ', child.text, [x for x in child.children], Gsent)
                self.proc_sc_phrase(child, start, end)  # treat sub. conjunction
                Onts = set(ont['w::' + child.text])
                feats = self.collectFeatures(Onts)
            elif dep not in ['det', ]:
                Onts = set(ont['w::' + child.text])
                feats = self.collectFeatures(Onts)
            totFeats = totFeats.union(set(feats))
        # print("OTHER RETURNING TOTFEATS: ", totFeats)
        return totFeats

    # as advmod as SCONJ[]
    # well advmod as SCONJ[]
    # as cc Cars
    # NOUN[as, well]
    # for token in doc:
    # print(token.text, token.tag_, token.dep_, token.head.text, token.head.pos_,[child for child in token.children])
    # in prep man NOUN [spite]
    # spite pobj in ADP [of]
    # of prep spite NOUN [warning]
    # that, if , as, because, for, of, since, before, like, after
    # Time:  after, as soon as, as long as, before,till, by the time, once, still, until, when, whenever, while ,now that
    # concession: although, as though, even though
    # comparison: just as, though, whereas, in contrast to, than, rather than, as much as, whether
    # cause/reason: as, because, in order to, in order that, since, so that, in order
    # condition: even if, if, in case, provided, provided that, unless, only if, assuming that, lest, supposing
    # place: where, wherever,
    # Manner: How, as though, as if
    # relative pronouns: who, whoever, whom, whomever, whose
    # why, where if,
    # def proc_sc_phrase(self, child, start, end):
    #  tag = child.text
    #     if tag in

    def proc_cc_phrase(self, child, start, end):  # am very suspicious - it may be catching spacy limitations
        # print("Entering CC",child.text)
        feats = set([])
        seq = [l for l in child.lefts]
        rights = [r for r in child.rights]
        # print(child.text, seq,rights)
        seq.append(child)
        # print(child.text, seq,rights)
        seq.extend(rights)
        # print(child.text, seq)
        phrase = [i.text for i in seq]
        #     if start > grandchild.left_edge.i:
        #         start = grandchild.left_edge.i
        #     if end < grandchild.right_edge.i + 1
        #         end = grandchild.right_edge.i + 1
        for grandchild in child.children:  # why child and grandchild
            if grandchild.dep_ != 'advmod' and ont['ont::' + child.text] != None:
                feats = feats.union(self.collectFeatures([ont['ont::' + child.text]]))
        if len(feats) > 0:
            return (feats, start, end)
        phrase = '_'.join(phrase)
        w_ont = OntWords[phrase]
        if w_ont not in [None, [], ()]:
            oval = [ont['ont::' + w_ont[0]]]
            if oval != None:
                feats = self.collectFeatures([ont['ont::' + w_ont[0]]])
                return (feats, self.SEtok[seq[0]][0], self.SEtok[seq[-1]][1])
        return (set([]), 0, 0)

    def proc_ont(self, wrd, start, end, pos=''):
        # print("Entering ONT ", wrd)
        if pos == '':
            otypes = ont['w::' + wrd]
        else:
            otypes = ont['w::' + wrd, convert_wn_to_trips(pos)]
        exp_otypes = set(otypes)
        totFeats = set([])
        # assertions = []
        for ext in otypes:  # goes up trips ontology
            exp_otypes.add(ext.parent)
        wnLemmas, wnNyms = self.WordToHypernymLemmas(wrd, pos)  # goes up wordnet hypernyms
        exp_otypes = set([y for w in wnLemmas for y in ont['w::' + w, convert_wn_to_trips(pos)]])
        if len(otypes) > 0:  # check
            o_featurelst = self.collectFeatures(otypes)
            totFeats = totFeats.union(o_featurelst)
            # assertion = self.pyke_assert(
            #    "({} {} {} {} {} {} {})".format('function', 'Ont', wrd, pos, o_featurelst, start, end), o_featurelst)
            # assertions.append(assertion)
        if len(exp_otypes) > 0:
            e_featurelst = self.collectFeatures(exp_otypes)
            totFeats = totFeats.union(e_featurelst)
            # assertion = self.pyke_assert(
            #    "({} {} {} {} {} {} {})".format('function', 'ExtOnt', wrd, pos, e_featurelst, start, end), e_featurelst)
            # assertions.append(assertion)
        # assertion = self.pyke_assert(
        #    "({} {} {} {} {} {} {})".format('function', 'Syn', wrd, pos, wnLemmas, start, end), wnLemmas)
        # assertions.append(assertion)

        totFeats = totFeats.union(wnLemmas)
        # print()
        # print("TOTAL ASSERTIONS : ", assertions)
        # print("WHAT ARE TOT FEATS?????? ", totFeats)
        # print()
        return totFeats

    def proc_by_verb(self, doc):
        verbs = set()
        for possible_subject in doc:
            if possible_subject.dep == 'nsubj' and possible_subject.head.pos == VERB:
                verbs.add(possible_subject.head)
        # print(verbs)

    def pyke_assert(self, assertion, feats=None):
        """makes pyke assertion into cerebella"""
        if 'function' in assertion:
            # print(">>>>>>>>>>>>>>>>> ASSERTION: ", assertion)
            alist = assertion.split(',')
            # print(">>>> ASSERTION returning: ", type(alist), alist)
            return set(assertion)

        # nvbg_engine.assert_(assertion)

    def step_semantic_analysis(self):
        """
        Runs the semantic analysis on the asserted knowledge - this assumes this code is inputing into the pyke
        version of cerebella to map functions to behaviors - ignore this
        """
        self.getEngine().activate('semantics_analyze')
        # Output the content of the WM in the log
        # logger.debug("WM content \n<%s>", self.retrieve_all_specific_facts('nvbg'))

    def generate(self, sent):
        """
        Main Function
        """
        global Gsent
        Gsent = sent
        # self.resetEngine()
        doc = nlp(sent)
        # print('\n>>>>>>>> SENTENCE: ', sent)
        seq = self.proc_parse(doc)
        if self.debug_print:
            dictprint(seq)
        return seq
        # self.inference()
        # self.procFunction()


cere = CII()


def get_categories(sent):
    try:
        return cere.generate(sent.rstrip())
    except Exception as e:
        print("could not get sentence categorization for sentence: ", sent)
        print(e)

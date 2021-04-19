
# this breaks up the phrases of a sentence into noun phrases using spacy
# it then uses wordnet's hypernyms to analyze noun's relation to conceptual categories
#      like group or process that correlate to gestures
# and similar_tos relatiions to analyze the modifier into metaphoric properties
#      so that modifiers like important can be tied to metaphors like size (big)
# the user can specify whatever conceptual categories and metaphoric properties
#      see metaphorlst and propertyAdjLst
# for this to be useful we would want some some emphasis either textually (very),
#      sentimet magnitude or prosodic emphasis
# an alternative to wordnet is word vectors but i also tried that see vector.py
#      and it didnt work well

# leeks word sense disambuiguation
# size has attribute big and little

# this specifies what conceptual categories the code is looking for
# these would mostly be categories that would be realized gesturally

# metaphorLst = ('group.n.01', 'concept.n.01', 'process.n.01')

filename = input("Filename: ")

try:
    tfPtr = open(filename, 'r')
except:
    tfPtr = open('tests.txt','r')



metaphorLst = (('group', ['group',], ('group.n.01',),
               {'big':('big','successful'),'small':('small','minor')}),
                #('group', ('group.n.01','group.n.02','group.n.07','group.n.08','group.n.09','group.n.06','group.n.15','group.n.20','group.n.23'),
                #{'big':('big','successful'),'small':('small','minor')}),
               # ('group', ('arrangement.n.02', 'arrangement.n.03', 'community.n.01', 'community.n.02', 'community.n.03',
               #            'community.n.04', 'community.n.05', 'community.n.06'),
               #  {'big': ('big', 'successful'), 'small': ('small', 'minor')}),
               ('cycle', ['iteration','looping','loop','performance'], ('repeat.n.01',), {'big':('big',),'forward':('forward',),'backward':('backward',),
                                           'fast':('fast',), 'slow':('slow',), 'small':('small',)}),
               ('concept', [], ('concept.n.01','plan.n.01',), {'big':('significant','big','good','successful'),
                                                           'small':('small','trivial')}),
               ('force', [], ('power.n.01','power.n.05','power.n.07'), {'strong':('strong','successful'),'weak':('small','weak')}),
               ('path', [], ('path.n.01','path.n.03','path.n.04'), {'straight':('straight',),'crooked':('crooked',), 'narrow':('narrow',),'wide':('wide',)}),
               ('process', ['increase','calculation','computation', 'computing','conversion','overcompensation','projection'], ('processing.n.01',), {'short': ('short',), 'long': ('long',), 'forward':('forward',),
                                              'backward':('backward',), 'good': ('good', 'sucessful'),
                                              'bad':('bad','unsucessful')}))
gestureRec = {'group':('frame','sweep'), 'cycle':('cycle',), 'concept':('frame','cup'), 'force':('push',),'path':('path',),'process':('cycle','path')}



wnVerblst = (('like', [], ('like.v.01',)),
    ('dislike', [], ('dislike.v.01',)),
    ('proceed', [], ('proceed.v.01','proceed.v.02','proceed.v.03','proceed.v.04','proceed.v.05','proceed.v.01',)),
    ('gather', ['gather','store'], ()),
    ('reject',[],()),

    )



# this specifies how conceptual modifiers that would be realized by altering the gesture

# propertyAdjLst = ('big', 'significant', 'small','fast','slow', 'succesful') # this identifies possible modifier categories that

import re
from nltk.corpus import wordnet as wn
# print(wn.synset('group.n.01'))
import spacy

from textblob import TextBlob

from pytrips.ontology import load as tload
ont = tload()
print(ont['ont::contain-reln'])
# print(ont["q::of",'p'])
print(ont["w::of"])
print(ont["w::audience"])

from pytrips.helpers import helpers as hlp
spacyTable=hlp.make_spacy_pos_table()

hypo = lambda s: s.hyponyms()
hyper = lambda s: s.hypernyms()



from spacy.symbols import nsubj, VERB, ADV, ADJ

from spacy.lemmatizer import Lemmatizer

from spacy import displacy

nlp = spacy.load("en_core_web_lg")


# Finding a verb with a subject from below — good
# verbs = set()
# for possible_subject in doc:
#     if possible_subject.dep == nsubj and possible_subject.head.pos == VERB:
#         verbs.add(possible_subject.head)
# print(verbs)


from nltk.wsd import lesk

def testLesk():
    toklst = ["Autonomous", "cars", "shift", "insurance", "liability", "toward", "manufacturers"]
    print('shift', toklst)
    print(lesk(toklst, 'shift', 'v'))
    print(lesk(toklst, 'shift', 'v').lemmas())
    print(lesk(toklst, 'shift', 'v').lemmas()[0])
    print(lesk(toklst, 'shift', 'v').lemmas()[0].key())
    print(lesk(toklst, 'shift', 'v').lemmas()[0].name())
    print(lesk(toklst, 'shift', 'v').lemmas()[0].count()) 
    # print(lesk(toklst, 'shift', 'v').troponym())

    print('hypernyms', list(lesk(toklst, 'shift', 'v').closure(hyper)))
    print('hypernyms', list(lesk(toklst, 'shift', 'v').lemmas()[0].hypernyms()))

    print('hyponyms', list(lesk(toklst, 'shift', 'v').closure(hypo)))
    print('hyponyms', list(lesk(toklst, 'shift', 'v').lemmas()[0].hyponyms()))

    print(lesk(toklst, 'shift', 'v').similar_tos())
    print(lesk(toklst, 'shift', 'v').lemmas()[0].similar_tos())

    print(lesk(toklst, 'shift', 'v').lemmas()[0].verb_groups())
    # print(lesk(toklst, 'shift', 'v').pertainnyms())
    print(lesk(toklst, 'shift', 'v').lemmas()[0].causes())

    print(wn.lemma_from_key(lesk(toklst, 'shift', 'v').lemmas()[0].key()))


# testLesk()


from pprint import pprint

# nltk.download('framenet_v17')
from nltk.corpus import verbnet as vn


verbnetlist = [('reject', ['throw-17.1', 'throw-17.1-1', 'throw-17.1-1-1', 'obtain-13.5.2']),
            ('aggregate', ['obtain-13.5.2', 'herd-47.5.2',]),
            ('social_process', ['correspond-36.1', ]),
            ('soft_request', ['order-60', ]),
            ('hard_request', ['order-60-1', ]),
            ('process', ['other_cos-45.4', ])]



vnDict = {}
def construct_vnlst():
    for typ, cls in (verbnetlist):
        vnSet = set([])
        # print(typ, ": ", cls)
        for c in (cls):
            vnSet = vnSet.union(vn.lemmas(c))
            for id in (vn.wordnetids(c)):
                # print(id, vn.getSenseNumber(id))
                # wn.synset('dog.n.1').lemmas()[0].key()
                # print(id)
                sense_key_regex = re.compile(r"(.*)\%(.*):(.*):(.*)")
                lemma, ss_type, _, lex_id = sense_key_regex.match(id).groups()
                ilex = int(lex_id) + 1
                slex_id = ("{:02d}".format(ilex))
                synset_id = ".".join([lemma, 'v', slex_id])
                try:
                    wnverbs = wn.synset(synset_id).lemma_names()
                    vnSet = vnSet.union(wn.synset(synset_id).lemma_names())
                except:
                    print('wrong: ',synset_id)
        vnDict[typ] = vnlst




# nlp = spacy.load("en_core_web_lg")

advDict = {}
#store adverb dictionary
with open("Adv.txt", 'r') as f:
    for line in f:
        items = line.split()
        advDict[items[0]] = items[1]


# def findVerbs(doc):
#     verbs = set()
#     for possible_subject in doc:
#         if possible_subject.dep == nsubj and possible_subject.head.pos == VERB:
#             verbs.add(possible_subject.head)
#     return(verbs)

def processHypernyms(lemma,pos):
    resSet =set([])
    for snse in wn.synsets(lemma,pos):
        resSet = resSet.union(snse.closure(hyper))
    return resSet

def processOnt(lemma,pos=None):
    if (pos != None):
        oDict = ont[("q::" + lemma, pos)]
    else:
        oDict = ont[("q::" + lemma)]
    Oset = set(oDict['lex']).union(oDict['wn'])
    resDict = {}
    for t_ont in Oset:
        resDict[t_ont] = {'parent':t_ont.parent, 'children':t_ont.children,'arguments':t_ont.arguments}
    return resDict


def processVerbVN(vrbLemma):
    cats = []
    for key in vnDict.keys():
        if vrbLemma in vnDict[key]:
            cats = cats.append(key)
    return cats



def processVerbTrips_I(vrbLemma):
    vDict = ont[("q::" + vrbLemma, 'v')]
    #print(vDict)
    Vset = set(vDict['lex']).union(vDict['wn'])
    #print(Vset)
    resDict = {}
    for t_ont in Vset:
        resDict[t_ont] = {'parent':t_ont.parent, 'children':t_ont.children,'arguments':t_ont.arguments}
    return resDict

def processVerbTrips_II(sentlst,vrbLemma):
    Vlist=list()
    for lemma in lesk(sentlst, vrbLemma, 'v').lemmas():
        wnOnt = ont[('wn::' + lemma.key())]
        Vlist.append(wnOnt)
    vDict ={}
    for ont_inst in Vlist:
        print(ont_inst)
        vDict[ont_inst] = {'parent':ont_inst.parent, 'children':ont_inst.children,'arguments':ont_inst.arguments}
    return vDict


def processVerbWN(vrbLemma):
    syns=wn.synsets(vrbLemma)
    
    return []

# def processPP(doc):

# def processSC(doc):



def processVerb(doc):
    vlst = findVerbs(doc)
    #use lesk to find sense?
    # try wordnet
    # try verbnet
    # try trips

    






# extract the noun phrases in a sentence
def extractNPs(chunks):
    lst = []
    for chunk in chunks:
        root = chunk.root.text
        # print(chunk.text.split())
        mods = chunk.text
        # print(root,mods)
        lst.append((root,mods))
    return lst


# the following collects synonyms of adj as a step to check with overlap with metaphoric properties
def collectSimilarToAdjs(wrd):
    biglst = set({})
    for syn in wn.synsets(wrd, 'a'): # create a list of words that are similar (synonyms) for the adjective big
        for l in syn.lemmas():
            biglst.add(l.name()) # using a list but really should use a set to simplify membership check below
        for syn in syn.similar_tos():
            for l in syn.lemmas():
                biglst.add(l.name())
    return biglst

# find the shortest hypernym path between a sense and some hypernym root conceptual category like process or group
def shortestPathSenseToHypernyms(syn,roots):
    if not(syn): return []
    grpdx = 100  # silly initialization
    pathlen = 101
    grppath = []
    groupHypernym = []
    for root in roots:  # for every every synonymm of word in sent (regardless of POS) assume a noun
        rootnym = wn.synset(root)
        paths = syn.hypernym_paths()
        if paths != None:
            for path in paths:  # check to see if that synonym has root as a hypernym
                if rootnym in path:
                    pathlen = len(path) - path.index(rootnym)  # what is distance to group.n.01
                    if pathlen < grpdx:  # across synonyms of wrd keep track of shortest hypernym path to root
                        grpdx = pathlen
                        grppath = path
                        groupHypernym = [syn, root, grpdx, grppath]  # store that shortest path
    return groupHypernym

# find the shortest hypernym path between a wordand some hypernym root conceptual category like process or group
def shortestPathWordToHypernyms(word,roots,pos):
    grpdx = 100  # silly initialization
    pathlen = 101
    grppath = []
    groupHypernym = []
    for syn in wn.synsets(word,pos):
        # print(syn,syn.hypernym_paths())
        for root in roots:
            rootnym = wn.synset(root)
            paths = syn.hypernym_paths()
            if paths != None:
                for path in paths:  # check to see if that synonym has root as a hypernym
                    if rootnym in path:
                        pathlen = len(path) - path.index(rootnym)  # what is distance to group.n.01
                        if pathlen < grpdx:  # across synonyms of wrd keep track of shortest hypernym path to root
                            grpdx = pathlen
                            grppath = path
                            groupHypernym = [syn, root, grpdx, grppath]  # store that shortest path
    return groupHypernym


# find the shortest hypernym path between a noun and some conceptual category like process or group
def shortestPathSense(wrd,root):
    grpdx = 100  # silly initialization
    pathlen = 101
    grppath = []
    groupHypernym = []
    rootnym = wn.synset(root)
    for syn in wn.synsets(wrd, 'n'):  # for every every synonymm of word in sent (regardless of POS) assume a noun
        for path in syn.hypernym_paths():  # check to see if that synonym has group.n.01 as a hypernym
            if rootnym in path:
                pathlen = len(path) - path.index(rootnym)  # what is distance to group.n.01
                if pathlen < grpdx:  # across synonyms of wrd keep track of shortest hypernym path to group.n.01
                    grpdx = pathlen
                    grppath = path
    if pathlen < 100: groupHypernym = [wrd, root, grpdx, grppath]  # store that shortest path
    return groupHypernym

def procSent(sent):
    sent = sent.replace('.',' .')
    sent = sent.replace(',',' ,')
    sent = sent.replace('!',' !')
    sent = sent.replace('?',' ?')
    sent = sent.replace(';',' ;')
    sent = sent.replace(':',' :')
    return sent

for sent in tfPtr: # check every sent (maybe incompleter) in a file - assumes one per line
    sent=sent.strip()
    bigAdjectives = []
    groupHypernym = []
    # print(sent)
    spacyDoc = nlp(sent) # prcess sentence using spacy
    displacy.serve(spacyDoc, style="dep")
    nchunks = spacyDoc.noun_chunks
    # NPlst = extractNPs(nchunks) # extract all the Noun Phrase
    for chunk in nchunks:    # process each noun phrase
        # print(mods)
        # if len(wrd) < 3: break

        grpdx = 100 # silly initialization
        pathlen = 101
        grppath = []
        groupHypernym = []
        # posDict ={}
        # swrd = lesk(procSent(sent), wrd, 'n')
        # swrd = lesk(mods, wrd, 'n')
        # print(lesk(sent, 'bank'))
        # print(swrd,mods)
        wrd = chunk.root.text
        mods = chunk.text.replace(wrd,'').split()


        for metaphor,seeds, senses,propertyAdjLst in metaphorLst:
            # find out if word (wrd) has some hypernym path to a metaphoric concept and keeptrack of shortest path
            # groupHypernym = shortestPathSenseToHypernyms(swrd, senses)
            groupHypernym = shortestPathWordToHypernyms(wrd, senses,'n')
            if len(groupHypernym) > 0:
                print("-> In Noun Phrase: ", chunk.text)
                print('   ->  {} has hypernym {} with dx {} and path {}'.format(*groupHypernym))
                # find out if modifiers in NP chunk are similar to one the modifiers in propertyAdjLst
                for mod in mods:
                    # posDict[token.text]=token.pos_
                    if mod in advDict: 
                        print('      ->> Adverb {} has property {}'.format(mod, advDict[mod]))
                    else:
                        for adj in propertyAdjLst:
                            adjSet = collectSimilarToAdjs(mod).intersection(collectSimilarToAdjs(adj))
                            if len(adjSet) > 0:  print('      ->> Modifier {} has property {} from properties {}'.format(mod,adj,adjSet))


    for token in spacyDoc:
        # print(token.text,token.tag_)
        try:
            s_pos = spacyTable[token.tag_]
        except:
            s_pos=token.tag_
        print('Last Step: ',token.text,token.pos_,s_pos)
        if (s_pos == 'v'):
            print(token.text)
            print(processOnt(token.lemma_, s_pos))
            print(token.lemma_, processVerbVN(token.text))
            print(token.lemma_, processVerbTrips_I(token.text))
            # print('aggregate', processVerbTrips_II(procSent(sent),'aggregate'))
            print(token.lemma_, processVerbWN(token.text))
        if (s_pos == 'r'):
            print(processOnt(token.lemma_, 'r'))
        if (s_pos == 'a'):
            print(processOnt(token.lemma_, 'a'))
        if (s_pos == 'n'):
            print(processOnt(token.lemma_, s_pos))
            # print(token.text, processVerbVN(token.text))
            # print(token.text, processVerbTrips_I(token.text))
            # # print('aggregate', processVerbTrips_II(procSent(sent),'aggregate'))
            # print(token.text, processVerbWN(token.text))
        if (s_pos == 'IN'):
            # print('Prep:')
            print(processOnt(token.lemma_))
            # print(token.lemma_, processVerbVN(token.text))
            # print(token.lemma_, processVerbTrips_I(token.text))
            # print('aggregate', processVerbTrips_II(procSent(sent),'aggregate'))
            # print(token.lemma_, processVerbWN(token.text))











                

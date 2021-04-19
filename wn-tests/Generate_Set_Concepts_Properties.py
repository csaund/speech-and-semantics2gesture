
from nltk.corpus import wordnet as wn


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
verblst = ()

advDict = {}
#store adverb dictionary
with open("Adv.txt", 'r') as f:
    for line in f:
        items = line.split()
        advDict[items[0]] = items[1]


hypo = lambda s: s.hyponyms()
hyper = lambda s: s.hypernyms()



# the following collects synonyms of adj as a step to check with overlap with metaphoric properties
def collectSimilarToAdjs(wrd):
    #print(wn.synsets(wrd,'a'))
    bigset = set([wrd])
    syns = wn.synsets(wrd,'a')
    ssyns = set(syns)
    for syn in syns:  # this step is risky
        ssyns.union(set(wn.synsets(syn.lemmas()[0].name(),'a')))
    for syn in ssyns: # create a set of words that are similar (synonyms) for the adjectives
        for l in syn.lemmas():
            bigset.add(l.name())
        # print('Hey0: ', bigset)
        for syn in syn.similar_tos():
            for l in syn.lemmas():
                # print(l.name())
                bigset.add(l.name())
    return bigset


# closures on nouns
def collecthyponyms(sense, depth):
    sense = wn.synset(sense)
    hypoSet = set([])
    hypo = lambda s: s.hyponyms()
    for ls in list(sense.closure(hypo, depth=depth)):
        for l in ls.lemmas():
            hypoSet.add(l.name())
    return hypoSet


#print(wn.synsets('small','s'))
#print(wn.synset('minor.a.10').lemmas())


for concept,seeds, senses,adjlst in metaphorLst:
    hyponymSet = set(seeds)
    for sense in senses:
        hyponymSet = hyponymSet.union(collecthyponyms(sense,6))
        #print('blaaaahh', hyponymSet)
    print('Current concept: ' + concept + '\n' + str(hyponymSet) + '\n')
    for property in adjlst.keys():
        adjset = {property}
        print('\n    Property: '+ property + '\n')
        for adj in adjlst[property]:
            adjset = adjset.union(collectSimilarToAdjs(adj))
        print("           Similar Terms:\n", adjset, '\n')






                

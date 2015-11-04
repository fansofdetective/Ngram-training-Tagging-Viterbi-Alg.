import sys
import nltk
import math

#this function takes the words from the training data and returns a python list of all of the words that occur more than 5 times
#wbrown is a python list where every element is a python list of the words of a particular sentence
def calc_known(wbrown):
    knownwords = []
    
    # this dictionary keeps the number of appearance for each single word
    cnt={}
    
    for sentence in wbrown:
        # in current list, go through every element(word) and do counting
        for i in range(len(sentence)):
            if sentence[i] in cnt:
                cnt[sentence[i]] +=1
            else:
                cnt[sentence[i]] = 1

    #go through cnt dictionary and look for knownwords
    for item in cnt:
        if(cnt[item]>5):
            knownwords.append(item)
    return knownwords

#this function takes a set of sentences and a set of words that should not be marked '_RARE_'
#brown is a python list where every element is a python list of the words of a particular sentence
#and outputs a version of the set of sentences with rare words marked '_RARE_'
def replace_rare(brown, knownwords):
    
    rare = []
    
    for sentence in brown:
        # sen_rare is a list of words and it will be element of list <rare>
        sen_rare=[]
        
        for i in range(len(sentence)):
            
            if sentence[i] in knownwords:
                sen_rare.append(sentence[i])
            
            else:
                sen_rare.append('_RARE_')
        rare.append(sen_rare)
    return rare

#this function takes the ouput from replace_rare and outputs it
def q3_output(rare):
    outfile = open("B3.txt", 'w')

    for sentence in rare:
        outfile.write(' '.join(sentence[2:-1]) + '\n')
    outfile.close()

#this function takes tags from the training data and calculates trigram probabilities
#tbrown (the list of tags) should be a python list where every element is a python list of the tags of a particular sentence
#it returns a python dictionary where the keys are tuples that represent the trigram, and the values are the log probability of that trigram
def calc_trigrams(tbrown):
    
    #this function is similar to the one in A1
    qvalues = {}
    bigram_n = {}
    trigram_n = {}
    tot_tup = 0
    
    for tags in tbrown:
        # tags is a list, go through this list once, we can finish the counting
        for i in range(len(tags)-2):
            #take 3 element as trigram and 2 element as bigram each time
            tri = (tags[i],tags[1+i],tags[2+i])
            bi = (tags[1+i],tags[2+i])
            if bi in bigram_n:
                bigram_n[bi] += 1
            else:
                bigram_n[bi] = 1
            if tri in trigram_n:
                trigram_n[tri] += 1
            else:
                trigram_n[tri] = 1

    for t in trigram_n:
        
        if (t[0],t[1]) != ('*', '*'):
            qvalues[t]=(trigram_n[t]+0.0)/bigram_n[(t[0],t[1])]
        
        else:
            qvalues[t]=(trigram_n[t]+0.0)/len(tbrown)
        
        qvalues[t]=math.log(qvalues[t],2)
    return qvalues

#this function takes output from calc_trigrams() and outputs it in the proper format
def q2_output(qvalues):
    #output
    outfile = open("B2.txt", "w")
    for trigram in qvalues:
        output = " ".join(['TRIGRAM', trigram[0], trigram[1], trigram[2], str(qvalues[trigram])])
        outfile.write(output + '\n')
    outfile.close()

#this function calculates emission probabilities and creates a list of possible tags
#the first return value is a python dictionary where each key is a tuple in which the first element is a word
#and the second is a tag and the value is the log probability of that word/tag pair
#and the second return value is a list of possible tags for this data set
#wbrown is a python list where each element is a python list of the words of a particular sentence
#tbrown is a python list where each element is a python list of the tags of a particular sentence
def calc_emission(wbrown, tbrown):
    
    evalues = {}
    
    taglist = []
    
    # need to keep track of the number of tags and word/tag pairs
    cnt_t={}
    cnt_wt={}
    
    # for each element in list_of_list:
    for i in range(len(wbrown)):
        for j in range(len(wbrown[i])):
            
            if tbrown[i][j] in cnt_t:
                cnt_t[tbrown[i][j]] += 1
            else:
                cnt_t[tbrown[i][j]] = 1
            
            if (wbrown[i][j],tbrown[i][j]) in cnt_wt:
                cnt_wt[(wbrown[i][j],tbrown[i][j])] += 1
            else:
                cnt_wt[(wbrown[i][j],tbrown[i][j])] = 1

    # use this function: p = c(w,t)/c(t) to calculate emission prob.
    for pair in cnt_wt:
        p = (cnt_wt[pair]+0.0)/cnt_t[pair[1]]
        p = math.log(p,2)
        
        #save to dictionary <evalues>. Keys are each pair of word,tag. Values are corresponding emission prob
        evalues[pair] = p

    # get tag_list
    taglist=cnt_t.keys()

    return evalues, taglist

#this function takes the output from calc_emissions() and outputs it
def q4_output(evalues):
    #output
    outfile = open("B4.txt", "w")
    for item in evalues:
        output = " ".join([item[0], item[1], str(evalues[item])])
        outfile.write(output + '\n')
    outfile.close()


#this function takes data to tag (brown), possible tags (taglist), a list of known words (knownwords),
#trigram probabilities (qvalues) and emission probabilities (evalues) and outputs a list where every element is a string of a
#sentence tagged in the WORD/TAG format
#brown is a list where every element is a list of words
#taglist is from the return of calc_emissions()
#knownwords is from the the return of calc_knownwords()
#qvalues is from the return of calc_trigrams
#evalues is from the return of calc_emissions()
#tagged is a list of tagged sentences in the format "WORD/TAG". Each sentence is a string with a terminal newline, not a list of tokens.
def viterbi(brown, taglist, knownwords, qvalues, evalues):
    tagged = []
    
    for tokens in brown:
        
        #initialization, both Pi and backpointer are dictionary
        pi={}
        backpointers={}
        
        #keep a copy of original sentence, will use to replace _RARE_
        cpy_tokens = list(tokens)
        
        # replace _RARE_ first, for tagging computation
        for k in range(len(tokens)):
            if tokens[k] not in knownwords:
                tokens[k] = '_RARE_'
    
        # special case: for first and second pair of word/tag, they are all */* */*
        pi[(0,'*','*')] = 0
        pi[(1,'*','*')] = 0
        
        
        # First trigram calculation
        for v in taglist:
            if ('*','*',v) in qvalues and (tokens[2],v) in evalues:
                if qvalues[('*','*',v)] + evalues[(tokens[2],v)] >= -1000.0:
                    p = qvalues[('*','*',v) ] + evalues[(tokens[2],v)]
                else:
                    p = -1000.0
            else:
                    p = -1000.0
    
            pi[(2,'*',v)] = p
            backpointers[(2, '*', v)] = '*'
        
        # Second trigram calculation
        for u in taglist:
            for v in taglist:
                if ('*', u, v) in qvalues and (tokens[3], v) in evalues and (2,'*', u) in pi:
                    if qvalues[('*', u, v)] + evalues[(tokens[3],v)] + pi[(2,'*',u)] >= -1000.0:
                        p = qvalues[('*', u, v)] + evalues[(tokens[3],v)] + pi[(2,'*',u)]
                        # if condition satisfied, (3,u,v)'s highest prob happens when '*' is the backpointer
                        backpointers[(3, u, v)] = '*'
                    else:
                        p = -1000.0
                        backpointers[(3, u, v)] = taglist[0]
                else:
                    p = -1000.0
                    backpointers[(3, u, v)] = taglist[0]

                pi[(3,u,v)]=p
    
    

        # From the first trigram without start mark to the end of tokens (except for the one includes STOP)
        for k in range(4,len(tokens)-1):
            
            for u in taglist:
                
                for v in taglist:
                    
                    p = -1000.0
                    
                    wtPair = (tokens[k], v)
                    
                    tmp_backp = taglist[0]
                    for w in taglist:
                        if (w,u,v) in qvalues and wtPair in evalues and (k-1,w,u) in pi:
                            if qvalues[(w, u, v)] + evalues[wtPair] + pi[(k-1,w,u)] >= p:
                                p = qvalues[(w, u, v)] + evalues[wtPair] + pi[(k-1,w,u)]
                                tmp_backp = w
                    backpointers[(k,u,v)] = tmp_backp
                    pi[(k,u,v)] = p
       
        # For the last trigram which contains 'STOP'
        for u in taglist:
            for v in taglist:
                if (u,v,'STOP') in qvalues:
                    if qvalues[(u, v, 'STOP')] + pi[(len(tokens)-2,u,v)] >= -1000.0:
                        p = qvalues[(u, v, 'STOP')] + pi[(len(tokens)-2,u,v)]
                        last3 = (u, v, 'STOP')
                    else:
                        p = -1000.0
                else:
                        p = -1000.0


        N = len(tokens)

        # create a list with length N(lenght of each tokens), to save corresponding tags for each words in tokens
        tg=[]
        for k in range (N):
            tg.append(None)
        

        for i in range(N-2):
            if i == 0 or i == 1 or i == 2:
                tg[i] = last3[2-i]
            else:
                tg[i] = backpointers[(N-i+1),tg[i-1],tg[i-2]]

        tmp_wt=''
        for k in range (N-3):
            tmp_wt += str(cpy_tokens[k+2])+ '/' + str(tg[N-3-k]) + ' '
        tmp_wt += '\n'

        tagged.append(tmp_wt)

    return tagged

#this function takes the output of viterbi() and outputs it
def q5_output(tagged):
    outfile = open('B5.txt', 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()




#this function uses nltk to create the taggers described in question 6
#brown is the data to be tagged
#tagged is a list of lists of tokens in the WORD/TAG format.
def nltk_tagger(brown):
    tagged = []
    
    # Create taggers according to assignment
    from nltk.corpus import brown as brown_training

    training = brown_training.tagged_sents(tagset='universal')

    unigram_tagger = nltk.UnigramTagger(training)
    bigram_tagger = nltk.BigramTagger(training)
    trigram_tagger = nltk.TrigramTagger(training)
    default_tagger= nltk.DefaultTagger('NOUN')
    bigram_tagger= nltk.BigramTagger(training, backoff=default_tagger)
    trigram_tagger= nltk.TrigramTagger(training, backoff=bigram_tagger)
    
    
    # Go through each sentence as a list of words
    for tokens in brown:
        tri_tag = trigram_tagger.tag(tokens)
        # exclude the starting marks
        tri_tag = tri_tag[2:-1]
        # list <readyPairs> is the element of tagged, and it contains the list of tagged pairs of word/tag
        readyPairs = []
        for pairs in tri_tag:
            readyPairs.append(pairs[0]+"/"+pairs[1])
        
        tagged.append(readyPairs)
    
    return tagged

def q6_output(tagged):
    outfile = open('B6.txt', 'w')
    for sentence in tagged:
        output = ' '.join(sentence) + '\n'
        outfile.write(output)
    outfile.close()

#a function that returns two lists, one of the brown data (words only) and another of the brown data (tags only)
def split_wordtags(brown_train):
    wbrown = []
    tbrown = []
    for sentence in brown_train:
        wtmp=[]
        ttmp=[]
        sentence = '*/* */* '+sentence+' STOP/STOP'
        words = sentence.split()
        for item in words:
            # find the highest index of '/' to avoid case like 1/2/NUM
            pnt = item.rfind('/')
            wtmp.append(item[:pnt])
            ttmp.append(item[pnt+1:])
        wbrown.append(wtmp)
        tbrown.append(ttmp)
    return wbrown, tbrown

def main():
    #open Brown training data
    infile = open("Brown_tagged_train.txt", "r")
    brown_train = infile.readlines()
    infile.close()
    
    #split words and tags, and add start and stop symbols (question 1)
    wbrown, tbrown = split_wordtags(brown_train)
    
    #calculate trigram probabilities (question 2)
    qvalues = calc_trigrams(tbrown)
    
    #question 2 output
    q2_output(qvalues)
    
    #calculate list of words with count > 5 (question 3)
    knownwords = calc_known(wbrown)
    
    #get a version of wbrown with rare words replace with '_RARE_' (question 3)
    wbrown_rare = replace_rare(wbrown, knownwords)
    
    #question 3 output
    q3_output(wbrown_rare)
    
    #calculate emission probabilities (question 4)
    evalues, taglist = calc_emission(wbrown_rare, tbrown)
    
    #question 4 output
    q4_output(evalues)
    
    #delete unneceessary data
    del brown_train
    del wbrown
    del tbrown
    del wbrown_rare
    
    #open Brown development data (question 5)
    infile = open("Brown_dev.txt", "r")
    brown_dev = infile.readlines()
    infile.close()
    
    #Format: transfer string to list of tokens
    brown_list = []
    for sentence in brown_dev:
        sentence = '* * '+ sentence+' STOP'
        tokens = nltk.word_tokenize(sentence)
        brown_list.append(tokens)
    
    #do viterbi on brown_dev (question 5)
    viterbi_tagged = viterbi(brown_list, taglist, knownwords, qvalues, evalues)

    #question 5 output
    q5_output(viterbi_tagged)

    brown_list = []
    for sentence in brown_dev:
        sentence = '* * '+ sentence+' STOP'
        tokens = nltk.word_tokenize(sentence)
        brown_list.append(tokens)
    #do nltk tagging here
    nltk_tagged = nltk_tagger(brown_list)
    
    #question 6 output
    q6_output(nltk_tagged)
if __name__ == "__main__": main()

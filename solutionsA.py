#a function that calculates unigram, bigram, and trigram probabilities
#brown is a python list of the sentences
#this function outputs three python dictionaries, where the key is a tuple expressing the ngram and the value is the log probability of that ngram
#make sure to return three separate lists: one for each ngram
import nltk
import math

def calc_probabilities(brown):
    unigram_p = {}
    unigram_n={}
    bigram_p = {}
    bigram_n = {}
    trigram_p = {}
    trigram_n = {}
    tot_tup = 0
    
    # Add symbles to the beginning/end of each sentence
    for sentence in brown:
        
        #toeknize sentence for each sentence in txt
        tokens=nltk.word_tokenize(sentence)
        
        #For unigram, only count STOP, so add STOP to list "tokens"
        tokens = tokens+['STOP']
        
        #unigram count
        for word in tokens:
            tot_tup += 1
            if word in unigram_n:
                unigram_n[word] += 1
            else:
                unigram_n[word] = 1
    
        #bigram token and tuples
        tokens = ['*'] + tokens
        bigram_tuples = tuple(nltk.bigrams(tokens))
        
        #bigram count
        for bigram_tuple in bigram_tuples:
            if bigram_tuple in bigram_n:
                bigram_n[bigram_tuple] += 1
            else:
                bigram_n[bigram_tuple] = 1
    
        #trigram tokens and tuples
        tokens = ['*'] + tokens
        trigram_tuples=tuple(nltk.trigrams(tokens))
        
        #trigram count
        for trigram_tuple in trigram_tuples:
            if trigram_tuple in trigram_n:
                trigram_n[trigram_tuple] += 1
            else:
                trigram_n[trigram_tuple] = 1

    #We've gone trhough whole brown txt, now calculate probabilities
    for uni in unigram_n:
        unigram_p[tuple([uni])] = math.log((unigram_n[uni]+0.0)/tot_tup, 2)
   
   
   # For tuples at the beginning of a sentence, divide the count by number of sentences; Else use count of tuple[0]
    for bi in bigram_n:
        if bi[0]!='*':
            bigram_p[bi]=(bigram_n[bi]+0.0)/unigram_n[bi[0]]
        else:
            bigram_p[bi]=(bigram_n[bi]+0.0)/len(brown)
        bigram_p[bi]= math.log(bigram_p[bi],2)

   # similar to bigram, the only difference is how we judge if current tuple is the beginning (here 2 *'s)
    for tri in trigram_n:
        if (tri[0],tri[1]) != ('*', '*'):
            trigram_p[tri]=(trigram_n[tri]+0.0)/bigram_n[(tri[0],tri[1])]
        else:
            trigram_p[tri]=(trigram_n[tri]+0.0)/len(brown)
        trigram_p[tri]=math.log(trigram_p[tri],2)

    return unigram_p, bigram_p, trigram_p



#each ngram is a python dictionary where keys are a tuple expressing the ngram, and the value is the log probability of that ngram
def q1_output(unigrams, bigrams, trigrams):
    #output probabilities
    outfile = open('A1.txt', 'w')
    for unigram in unigrams:
        outfile.write('UNIGRAM ' + unigram[0] + ' ' + str(unigrams[unigram]) + '\n')
    for bigram in bigrams:
        outfile.write('BIGRAM ' + bigram[0] + ' ' + bigram[1]  + ' ' + str(bigrams[bigram]) + '\n')
    for trigram in trigrams:
        outfile.write('TRIGRAM ' + trigram[0] + ' ' + trigram[1] + ' ' + trigram[2] + ' ' + str(trigrams[trigram]) + '\n')
    outfile.close()

#a function that calculates scores for every sentence
#ngram_p is the python dictionary of probabilities
#n is the size of the ngram
#data is the set of sentences to score
#this function must return a python list of scores, where the first element is the score of the first sentence, etc.
def score(ngram_p, n, data):
    scores = []
    for sentence in data:
        tmp=0
        words = nltk.word_tokenize(sentence)
        words = words + ['STOP']
        
        # unigram
        if n==1:
            for word in words:
                uni = tuple([word])
                tmp += ngram_p[uni]
            scores.append(tmp)
    
        #bigram
        if n == 2:
            words = ['*']+ words
            bigrams=tuple(nltk.bigrams(words))
            for bi in bigrams:
                tmp=tmp+ngram_p[bi]
            scores.append(tmp)
    
        #trigram
        if n == 3:
            words = ['*'] + ['*'] + words
            trigrams = tuple(nltk.trigrams(words))
            for tri in trigrams:
                tmp=tmp+ngram_p[tri]
            scores.append(tmp)
    return scores



#this function outputs the score output of score()
#scores is a python list of scores, and filename is the output file name
def score_output(scores, filename):
    outfile = open(filename, 'w')
    for score in scores:
        outfile.write(str(score) + '\n')
    outfile.close()



#this function scores brown data with a linearly interpolated model
#each ngram argument is a python dictionary where the keys are tuples that express an ngram and the value is the log probability of that ngram
#like score(), this function returns a python list of scores

def linearscore(unigrams, bigrams, trigrams, brown):
    scores = []
    for sentence in brown:
        probability_inter = 0.0
        p=0.0
        flag = 0
        words = nltk.word_tokenize(sentence)
        words = ['*'] + ['*']+ words + ['STOP']
        
        #len(words)-2 is the number of tuples for all three ngrams
        for i in range(len(words)-2):
            
            #unigram counts from the 3rd position for each sentence(ignoring * *)
            uni = tuple([words[i+2]])
            
            #bigram ignores the first * for each sentence
            bi = (words[i+1],words[i+2])
            
            #trigram fetches 3 tokens each time, right from the beginning of each sentences
            tri = (words[i],words[i+1],words[i+2])
            
            #calculate 3 probabilities by doing exponential. If OOV, we flag it and set the whole sentence_probability as -1000
            if uni in unigrams:
                p_u=2**unigrams[uni]
            else:
                flag = 1
                break
            if bi in bigrams:
                p_b=2**bigrams[bi]
            else:
                flag = 1
                break
            if tri in trigrams:
                p_t=2**trigrams[tri]
            else:
                flag = 1
                break
            p+=math.log((p_u + p_b + p_t),2) - math.log(3,2)
        if flag:
            p = -1000
        scores.append(p)
    return scores




def main():
    #open data
    infile = open('Brown_train.txt', 'r')
    brown = infile.readlines()
    infile.close()

    #calculate ngram probabilities (question 1)
    unigrams, bigrams, trigrams = calc_probabilities(brown)

    #question 1 output
    q1_output(unigrams, bigrams, trigrams)

    #score sentences (question 2)
    uniscores = score(unigrams, 1, brown)
    biscores = score(bigrams, 2, brown)
    triscores = score(trigrams, 3, brown)

    #question 2 output
    score_output(uniscores, 'A2.uni.txt')
    score_output(biscores, 'A2.bi.txt')
    score_output(triscores, 'A2.tri.txt')

    #linear interpolation (question 3)
    linearscores = linearscore(unigrams, bigrams, trigrams, brown)

    #question 3 output
    score_output(linearscores, 'A3.txt')

    #open Sample1 and Sample2 (question 5)
    infile = open('Sample1.txt', 'r')
    sample1 = infile.readlines()
    infile.close()
    infile = open('Sample2.txt', 'r')
    sample2 = infile.readlines()
    infile.close() 

    #score the samples
    sample1scores = linearscore(unigrams, bigrams, trigrams, sample1)
    sample2scores = linearscore(unigrams, bigrams, trigrams, sample2)

    #question 5 output
    score_output(sample1scores, 'Sample1_scored.txt')
    score_output(sample2scores, 'Sample2_scored.txt')

if __name__ == "__main__": main()

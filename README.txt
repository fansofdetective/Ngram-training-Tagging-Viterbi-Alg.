{\rtf1\ansi\ansicpg1252\cocoartf1344\cocoasubrtf720
{\fonttbl\f0\fswiss\fcharset0 Helvetica;\f1\fnil\fcharset0 Menlo-Regular;}
{\colortbl;\red255\green255\blue255;}
\margl1440\margr1440\vieww16980\viewh11140\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural

\f0\fs24 \cf0 \
\
\
xl2454        Xiaojing Liu\
\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\
Running time:\
	Part A: 2~3 mins\
	Part B: 10 mins\
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural
\cf0 \
\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural
\cf0 \
\
A2: Perplexity of unigram, bigram and trigram:\
	Unigram: 
\f1\fs22 \CocoaLigature0 1104.83292814\
	Bigram: 57.2215464238\
	Trigram: 5.89521267642\
\
\
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural

\f0\fs24 \cf0 \CocoaLigature1 \
\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural

\f1\fs22 \cf0 \CocoaLigature0 \
\
A4: \
\
In textbook we know the smaller the perplexity is, the higher is the model performance, the model will be less \'93surprised\'94 by text data. The perplexity of model with linear interpolation is: 13.0759217039. Compared with unigram model and bigram models, this model performs much better. Although this model\'92s perplexity is higher than trigram model, it is more flexible. Combining unigram, bigram and trigram, linear interpolation model will perform stably even when in face with a test set where trigrams are quite different.\
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural

\f0\fs24 \cf0 \CocoaLigature1 \
\
\
\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural

\f1\fs22 \cf0 \CocoaLigature0 \
A5: Perplexity of Sample1.txt: 11.6492786046\
    Perplexity of Sample2.txt: 1627571078.54\
	Apparently, Sample2.txt doesn\'92t belong to Brown dataset, while Sample1.txt belongs to this dataset. From perplexity we see, Sample1\'92s perplexity is very low while Sample2\'92s perplexity is ridiculously high. This means in Sample2, most of the n-grams are never seen in training dataset, which is Brown dataset. \
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural

\f0\fs24 \cf0 \CocoaLigature1 \
\
\
\
\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural

\f1\fs22 \cf0 \CocoaLigature0 \
B5: Percent correct tags of B5.txt: 92.0860102522\
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural

\f0\fs24 \cf0 \CocoaLigature1 \
\
\
\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural

\f1\fs22 \cf0 \CocoaLigature0 \
B6: Percent correct tag of B6.txt: 95.3123637315. This is higher than the one of B5, which used HMM for tagging. Under this dataset from Brown Corpus, NLTK tagger performs better than HMM tagger.\
\
\
\
\
\
\
}
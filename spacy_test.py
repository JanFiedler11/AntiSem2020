import spacy

nlp=spacy.load('en_core_web_sm')

text=('A letter has been written, asking him to be released')

my_doc=nlp(u"A letter has been written, asking him to be released")

for token in my_doc:
    print(token.text,token.is_stop,token.lemma_)

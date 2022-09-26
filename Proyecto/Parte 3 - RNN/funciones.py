import spacy
def load_glove_model(glove_file):
    print("[INFO]Cargando GloVe Model...")
    model = {}
    with open(glove_file, 'r') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embeddings = [float(val) for val in split_line[1:]]
            model[word] = embeddings
    print("[INFO] descargado!".format(len(model)))
    return model
# adopted from utils.py
nlp = spacy.load("en")

def remove_stopwords(sentence):
    '''
    Remover stopwords
    '''
    new = []
    # tokenizar
    sentence = nlp(sentence)
    for tk in sentence:
        if (tk.is_stop == False) & (tk.pos_ !="PUNCT"):
            new.append(tk.string.strip())
    # covertir a string
    c = " ".join(str(x) for x in new)
    return c

def lemmatize(sentence):
    '''
    lematizacion
    '''
    sentence = nlp(sentence)
    s = ""
    for w in sentence:
        s +=" "+w.lemma_
    return nlp(s)

def sent_vectorizer(sent, model):
    '''
    vectorizar
    '''
    sent_vector = np.zeros(200)
    num_w = 0
    for w in sent.split():
        try:
            # add up all token vectors to a sent_vector
            sent_vector = np.add(sent_vector, model[str(w)])
            num_w += 1
        except:
            pass
    return sent_vector


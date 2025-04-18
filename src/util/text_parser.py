import spacy

nlp = spacy.load('en_core_web_sm')

def process_text(sentence):
    sentence = sentence.replace('-', '')
    doc = nlp(sentence)
    word_list = []
    pos_list  = []
    for token in doc:
        word = token.text
        if not word.isalpha():        # 알파벳 아닌 토큰은 건너뛰기
            continue
        # NOUN/VERB 원형(lemma) 사용
        if (token.pos_ in ('NOUN','VERB')) and (word.lower() != 'left'):
            word_list.append(token.lemma_)
        else:
            word_list.append(word)
        pos_list.append(token.pos_)
    return word_list, pos_list

#원문 --> 토큰붙은 문장 변환기 
def encode_tagged(sentence):
    words, poses = process_text(sentence)
    token_str = ' '.join(f"{w}/{p}" for w, p in zip(words, poses))
    return token_str

#토큰붙은 문장 --> 원문 변환기
def decode_tagged(encoded_sentence):
    tokens = encoded_sentence.strip().split()
    
    words = []
    for tok in tokens:
        if "/" in tok:
            word, _pos = tok.rsplit("/", 1)
            words.append(word)
    return " ".join(words)
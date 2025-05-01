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

def split_by_first_hash(text, split_char='#'):
    index = text.find('#')  # 첫 번째 #의 인덱스 찾기
    if index != -1:
        return text[:index], text[index:]  # # 포함해서 뒤쪽에 붙이기
    else:
        return text, ''  # #이 없는 경우

#원문 --> 토큰붙은 문장 변환기 
def encode_tagged(sentence):
    lines = sentence.splitlines()
    sentences = ""
    for line in lines:
        line, timetag = split_by_first_hash(line)
        print(f'라인: {line}, 타임태그: {timetag}')
        words, poses = process_text(line)
        token_str = ' '.join(f"{w}/{p}" for w, p in zip(words, poses))
        sentence_in_line = line + " " + token_str
        if timetag:
            sentence_in_line += f"{timetag.strip()}"
        else:
            sentence_in_line += "#0.0#0.0"
        sentences += sentence_in_line.strip() + "\n"
        print(f"토큰화된 문장: {sentence_in_line.strip()}")
    
    print(f"인코딩된 문장: {sentences}")
    return sentences

#토큰붙은 문장 --> 원문 변환기
def decode_tagged(encoded_sentence):
    lines = encoded_sentence.splitlines()
    tokens = []
    for line in lines:
        line, timetag = split_by_first_hash(line)
        line = " ".join(part for part in line.split() if "/" not in part)
        line = ''.join(char for char in line if char not in ['!', '?', ',', ':', ';'])
        tokens.append(f'{line.strip()}{timetag.strip()}\n')
    
    print(f"{encoded_sentence} -> 디코딩된 토큰: {tokens}")
    return ''.join(tokens)

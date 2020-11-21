import random

import spacy
from hazm import Normalizer, Stemmer, WordTokenizer, sent_tokenize


def train_spacy(data, iterations):
    TRAIN_DATA = data
    nlp = spacy.blank('fa')
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)

    for _, annotations in TRAIN_DATA:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        for itn in range(iterations):
            print("Statring iteration " + str(itn))
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                nlp.update(
                    [text],
                    [annotations],
                    drop=0.2,
                    sgd=optimizer,
                    losses=losses)
            print(losses)
    return nlp


def prepare():
    normalizer = Normalizer()
    stemmer = Stemmer()

    string = '''ویکی پدیای انگلیسی در تاریخ ۱۵ ژانویه ۲۰۰۱ (۲۶ دی ۱۳۷۹) به صورت مکملی برای دانشنامهٔ تخصصی نیوپدیا نوشته شد. بنیان گذاران آن «جیمی ویلز» و «لری سنگر» هستند. هم اکنون بنیاد غیرانتفاعی ویکی مدیا پروژهٔ ویکی پدیا را پشتیبانی می کند. میزبان های اینترنتی اصلی این وبگاه در شهر تامپای فلوریدا هستند؟ همچنین میزبان های اضافی دیگری هم در شهرهای آمستردام، شیراز و سئول به این وبگاه یاری می رسانند؟'''

    tokenizer = WordTokenizer(join_verb_parts=True, separate_emoji=True, replace_links=True, replace_IDs=True,
                              replace_emails=True, replace_numbers=True, replace_hashtags=True)

    labels = {'،': 'COMMA',
              '.': 'DOT',
              '؟': 'QMARK'}
    normal_string = normalizer.normalize(string)
    for label in labels.keys():
        print(normal_string.find(label))

    exit(0)
    for i, sent in enumerate([1, 2, 3, 4]):
        entities = []
        (10, 15, 'PrdName')
        for label in labels.keys():
            print(f'{label} in {i}', label in sent)
        record = (sent, {'entities': entities})

        print()


prepare()

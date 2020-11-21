import torch
import random
import pandas as pd
import spacy
from hazm import Normalizer, Stemmer, WordTokenizer, sent_tokenize
from transformers import (AutoConfig, AutoModel, AutoTokenizer)
from .utils_ner import TokenClassificationDataset, Split


def prepare(input_string):
    special_labels = {'،': 'COMMA',
              '.': 'DOT',
              '؟': 'QMARK'}
    tokenizer = WordTokenizer(join_verb_parts=True, separate_emoji=True, replace_links=True, replace_IDs=True,
                              replace_emails=True, replace_numbers=True, replace_hashtags=True)
    normalizer = Normalizer()
    normal_string = normalizer.normalize(input_string)
    list_of_tuples = []
    tokenized = tokenizer.tokenize(normal_string)
    for i, token in enumerate(tokenized):
        lbl = 'O'
        brek = False
        for label in special_labels.keys():
            if token == label:
                if list_of_tuples:
                    list_of_tuples[-1][1] = special_labels[label]
                    brek = True
                    break
        if not brek:
            list_of_tuples.append([token, lbl])
    return list_of_tuples


string = '''
سلام ویکی پدیای انگلیسی در تاریخ ۱۵ ژانویه ۲۰۰۱ (۲۶ دی ۱۳۷۹) به صورت مکملی برای دانشنامهٔ تخصصی نیوپدیا نوشته شد.
بنیان گذاران آن «جیمی ویلز» و «لری سنگر» هستند.
هم اکنون بنیاد غیرانتفاعی ویکی مدیا پروژهٔ ویکی پدیا را پشتیبانی می کند.
میزبان های اینترنتی اصلی این وبگاه در شهر تامپای فلوریدا هستند؟ همچنین میزبان های اضافی دیگری هم در شهرهای آمستردام،
شیراز و سئول به این وبگاه یاری می رسانند.'''
# l_o_t = prepare(string)
# train_df = pd.DataFrame(l_o_t, columns=['words', 'labels'])
# print(train_df.tail())


with open("../data/Persian-WikiText-1.txt", 'r') as reader:
    spc = reader.readlines()
    filtered = [i for i in spc if len(i) > 50]

# test_df = pd.DataFrame(test_tuples, columns=['sentence_id', 'words', 'labels'])

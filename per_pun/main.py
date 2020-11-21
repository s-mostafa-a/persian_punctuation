import spacy
import random

TRAIN_DATA = [('سلام خوبی آشغال؟', {'entities': [(10, 15, 'PrdName')]}),
              ('سلام خوبی احمق روانی؟', {'entities': [(10, 20, 'PrdName')]}),
              ('سلام خوبی گاو توله؟', {'entities': [(10, 20, 'PrdName')]})]
              # ('what is the price of jegging?', {'entities': [(21, 28, 'PrdName')]}),
              # ('what is the price of t-shirt?', {'entities': [(21, 28, 'PrdName')]}),
              # ('what is the price of jeans?', {'entities': [(21, 26, 'PrdName')]}),
              # ('what is the price of bat?', {'entities': [(21, 24, 'PrdName')]}),
              # ('what is the price of shirt?', {'entities': [(21, 26, 'PrdName')]}),
              # ('what is the price of bag?', {'entities': [(21, 24, 'PrdName')]}),
              # ('what is the price of cup?', {'entities': [(21, 24, 'PrdName')]}),
              # ('what is the price of jug?', {'entities': [(21, 24, 'PrdName')]}),
              # ('what is the price of plate?', {'entities': [(21, 26, 'PrdName')]}),
              # ('what is the price of glass?', {'entities': [(21, 26, 'PrdName')]}),
              # ('what is the price of moniter?', {'entities': [(21, 28, 'PrdName')]}),
              # ('what is the price of desktop?', {'entities': [(21, 28, 'PrdName')]}),
              # ('what is the price of bottle?', {'entities': [(21, 27, 'PrdName')]}),
              # ('what is the price of mouse?', {'entities': [(21, 26, 'PrdName')]}),
              # ('what is the price of keyboad?', {'entities': [(21, 28, 'PrdName')]}),
              # ('what is the price of chair?', {'entities': [(21, 26, 'PrdName')]}),
              # ('what is the price of table?', {'entities': [(21, 26, 'PrdName')]}),
              # ('what is the price of watch?', {'entities': [(21, 26, 'PrdName')]})]


def train_spacy(data, iterations):
    TRAIN_DATA = data
    nlp = spacy.blank('fa')  # create blank Language class
    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)

    # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(iterations):
            print("Statring iteration " + str(itn))
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                nlp.update(
                    [text],  # batch of texts
                    [annotations],  # batch of annotations
                    drop=0.2,  # dropout - make it harder to memorise data
                    sgd=optimizer,  # callable to update weights
                    losses=losses)
            print(losses)
    return nlp


prdnlp = train_spacy(TRAIN_DATA, 20)

# Save our trained Model
modelfile = input("Enter your Model Name: ")
prdnlp.to_disk(modelfile)

# Test your text
test_text = input("Enter your testing text: ")
doc = prdnlp(test_text)
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)
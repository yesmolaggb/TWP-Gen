import spacy

nlp = spacy.load("en_core_web_lg")
doc = nlp("Even as the don't U.K. secretary of homeland security was putting his people on high alert last month, a 30-foot Cuban patrol boat with four heavily armed men landed on American shores, utterly undetected by the Coast Guard Secretary Ridge now leads.")

for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_, "({})".format(spacy.explain(ent.label_)))
print("\n")
for token in doc:
    print(token.text, token.dep_, token.head.text, token.head.pos_,
            [child for child in token.children])
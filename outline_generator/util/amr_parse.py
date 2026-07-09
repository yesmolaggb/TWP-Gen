import spacy
import amrlib

# From: https://spacy.io/universe/project/amrlib/
# Install: https://amrlib.readthedocs.io/en/latest/install/

amrlib.setup_spacy_extension()
nlp = spacy.load('en_core_web_sm')
doc = nlp("Even as the secretary of homeland security was putting his people on high alert last month, a 30-foot Cuban patrol boat with four heavily armed men landed on American shores, utterly undetected by the Coast Guard Secretary Ridge now leads.")
# doc = nlp("Paul, as I understand your definition of a political -- of a professional politician based on that is somebody who is elected to public office.")
graphs = doc._.to_amr()
for graph in graphs:
    print(graph)
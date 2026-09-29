# Verb-sense dictionary construction

- `crawl_verb_senses.py` identifies candidate verbs, retrieves sense definitions,
  and retains verbs with multiple senses.
- `add_examples.py` generates additional examples for each retained sense with a
  locally deployed language model.

Configure paths with environment variables:

```bash
export TWPGEN_WORD_LIST=/path/to/words.txt
export TWPGEN_VERB_DICT_OUTPUT=/path/to/verb_sense_dict_zh.json
export TWPGEN_DICTIONARY_MODEL=/path/to/local/model
export TWPGEN_VERB_DICT_INPUT=/path/to/verb_sense_dict_zh.json
export TWPGEN_VERB_DICT_WITH_EXAMPLES=/path/to/verb_sense_dict_with_examples.json
```

Run the two stages in order:

```bash
python experiments/dictionary/crawl_verb_senses.py
python experiments/dictionary/add_examples.py
```

The source word lists and generated dictionary are not committed here. Users
should ensure that their source data and web access comply with the applicable
licenses and terms of service.

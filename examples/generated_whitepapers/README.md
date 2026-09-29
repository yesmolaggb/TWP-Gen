# Generated examples

This directory provides 10 representative generated white papers, their
corresponding citation records, and their document-level evaluation scores.
The article and reference file with the same stem form one complete example.

- `articles/`: generated white papers.
- `references/`: citation and source records paired with the articles.
- `scores.json`: the ten metric scores, three group means, and overall mean for
  every published example.

The scores were produced in one completed evaluation run using
Qwen3-32B-Instruct as the evaluator. Every metric is reported on the 0--5 scale
defined in `dataset/evaluation/`. Group means and the overall score are
unweighted arithmetic means. These example-level scores support inspection of
the released samples; they do not replace the paper's aggregate results over
all 60 topics and repeated evaluation runs.

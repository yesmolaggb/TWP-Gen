# Section-taxonomy construction

This directory contains the programs used to derive the 11 technical-white-paper
section types from the table-of-contents corpus described in the paper.

## Pipeline

1. `extract_toc_candidates.py`: extracts candidate TOC lines from OCR Markdown.
2. `reconstruct_toc.py`: reconstructs a normalized two-level TOC.
3. `filter_toc.py`: previews TOC quality and retention decisions.
4. `parse_toc.py`: converts every retained TOC into `toc_table.csv`.
5. `classify_toc_sections.py`: assigns first-level headings to the 11 section types.
6. `analyze_categories.py`: computes category occurrence, position, and depth statistics.
7. `analyze_sequences.py`: analyzes section sequences and common templates.

## Configuration

The original absolute server paths and embedded credentials have been removed.
Configure the programs with environment variables:

```bash
export TWPGEN_TAXONOMY_OCR_DIR=/path/to/paddle_ocr_output
export TWPGEN_TAXONOMY_DATA_DIR=/path/to/taxonomy_workspace
export DASHSCOPE_API_KEY=your_key
export DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export TWPGEN_TAXONOMY_MODEL=qwen3-max
```

The source white papers are not redistributed because they remain the property
of their publishers. The scripts can be applied to OCR output with the same
directory structure.

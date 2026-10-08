#!/usr/bin/env python3
"""Normalize the prepared DuEE labels and export the controlled-subset manifest."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np


LABEL_BY_COUNT = {
    408: "竞赛行为-胜负",
    179: "产品行为-发布",
    144: "司法行为-拘捕",
    79: "组织关系-加盟",
    33: "组织关系-辞/离职",
    32: "人生-死亡",
    26: "灾害/意外-车祸",
    22: "司法行为-起诉",
}


def recover_label(raw_label: str, count: int) -> str:
    if count in LABEL_BY_COUNT:
        return LABEL_BY_COUNT[count]
    if count == 44 and (raw_label.startswith("司") or raw_label.startswith("˾")):
        return "司法行为-约谈"
    if count == 44:
        return "组织关系-退出"
    raise ValueError(f"Unexpected label/count pair: {raw_label!r}, {count}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_npz", type=Path)
    parser.add_argument("output_npz", type=Path)
    parser.add_argument("output_manifest", type=Path)
    args = parser.parse_args()

    arrays = dict(np.load(args.input_npz, allow_pickle=False))
    counts = Counter(arrays["labels"].tolist())
    label_map = {
        raw_label: recover_label(raw_label, count)
        for raw_label, count in counts.items()
    }
    clean_labels = np.asarray([label_map[label] for label in arrays["labels"]])
    arrays["labels"] = clean_labels

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **arrays)

    label_names = sorted(set(clean_labels.tolist()))
    label_ids = {label: index for index, label in enumerate(label_names)}
    with args.output_manifest.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("row_index", "sentence_id", "label_id", "label"))
        for row_index, sentence_id, label in zip(
            arrays["row_indices"], arrays["sentence_ids"], clean_labels
        ):
            writer.writerow((int(row_index), sentence_id, label_ids[label], label))

    print(f"instances={len(clean_labels)} classes={len(label_names)}")


if __name__ == "__main__":
    main()

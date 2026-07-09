import pickle as pk
import twpgen_config as args

from tqdm import tqdm

from knowledge_graph_client import ConceptNetRequest


def main(filepath: str, savepath: str, weight_threshold: int, weight_v2n: int) -> None:
    parsed_corpus_po_mention = []

    word_list = []
    with open(filepath, "rb") as f:
        parsed_corpus_po_mention = pk.load(f)
    pre_sent_id = -1
    for svo_id, corpus_info in tqdm(parsed_corpus_po_mention.items()):
        verb = corpus_info["verb"]
        obj_head = corpus_info["obj_head"]
        entitys_embed = corpus_info["entitys_embed"]
        [sent_id, vo_id] = svo_id.split("_")
        sent_id = int(sent_id)
        vo_id = int(vo_id)
        word_list.append(
            {
                "word": verb,
                "sent_id": sent_id,
                "vo_id": vo_id,
                "word_type": "v",
            }
        )
        if obj_head != None and obj_head != "":
            word_list.append(
                {"word": obj_head, "sent_id": sent_id, "vo_id": vo_id, "word_type": "o"}
            )
        if sent_id != pre_sent_id:
            pre_sent_id = sent_id
            # one setence may have multi vo pair, but entity is repeat, so just append once
            for entity in entitys_embed:
                ent_name = entity["ent_name"].lower()
                ent_name = "_".join(ent_name.split(" "))
                ent_label = entity["ent_label"]
                word_list.append(
                    {
                        "word": ent_name,
                        "sent_id": sent_id,
                        "vo_id": vo_id,
                        "word_type": ent_label,
                    }
                )
    cnet = ConceptNetRequest(
        word_list=word_list,
        weight_threshold=weight_threshold,
        weight_v2n=weight_v2n,
        save_path=savepath,
    )
    print("Request ConceptNet WebAPI and Building Word Graph...")
    cnet.request()


if __name__ == "__main__":
    main(
        filepath=args.mention_file,
        savepath=args.graph_path,
        weight_threshold=0,
        weight_v2n=9,
    )

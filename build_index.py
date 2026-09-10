import pandas as pd
import json
import numpy as np
from config import *

def read_preprocess(base:Path): 
    df=pd.read_parquet(base / "tokens.parquet")
    df["tokens"] = [
        list(h) + list(b)
        for h, b in zip(df["headline_tokens"], df["body_tokens"])
    ]
    # 按tokens列表拆成多行
    long = df[["news_id", "tokens"]].explode("tokens", ignore_index=True)
    # 按文档分组，依次标记位置
    long["pos"] = long.groupby("news_id", sort=False).cumcount()

    long = long.rename(columns={"tokens": "term"})
    # 将原新闻编号映射成从0开始的编号，并形成一张映射表
    nid_to_int = {nid: i for i, nid in enumerate(df["news_id"])}
    long["doc_id"] = long["news_id"].map(nid_to_int).astype("int64")
    # 按词项字典序、文档编号、词项位置排序
    long_sorted = long.sort_values(["term", "doc_id", "pos"], ignore_index=True)

    return long_sorted,nid_to_int

class PostingList:
    def __init__(self,doc_ids:list[int],positions:list[list]):
        self.doc_ids=doc_ids
        self.positions=positions
        self.tfs = [len(p) for p in positions]

def build_idx(long_sorted:pd.DataFrame):
    word_idx = {}   # term -> PostingList
    cur_term = None
    cur_did = None
    cur_positions = []
    doc_ids = []
    positions = []

    for term, doc_id, pos in zip(
        long_sorted["term"], long_sorted["doc_id"], long_sorted["pos"]
    ):
        if term!=cur_term:
            if cur_term is not None:
                doc_ids.append(cur_did)
                positions.append(cur_positions)
                word_idx[cur_term] = PostingList(doc_ids,positions)
            cur_term=term
            cur_did = doc_id
            cur_positions = [pos]
            doc_ids = []
            positions = []
        elif doc_id!=cur_did:
            if cur_did is not None:
                doc_ids.append(cur_did)
                positions.append(cur_positions)
            cur_did=doc_id
            cur_positions=[pos]
        else:
            cur_positions.append(pos)

    if cur_term is not None:
        doc_ids.append(cur_did)
        positions.append(cur_positions)
        word_idx[cur_term] = PostingList(doc_ids,positions)
    return word_idx

# 写入磁盘
def save_to_disk(base:Path,word_idx:dict,nid_to_int:dict):
    terms = sorted(word_idx.keys())            # 排序让 offset 好算
    term_to_id = {t: i for i, t in enumerate(terms)}

    all_doc_ids = []      # 全局 doc_id 流
    all_tfs = []          # 与 doc_id 一一对应的 tf
    all_positions = []    # 所有 doc 的 positions 拼接
    pos_offsets = [0]     # 每个 posting 的 positions 在 all_positions 里的起始
    term_meta = {}        # term -> {tid, df, offset}

    for tid, t in enumerate(terms):
        pl = word_idx[t]
        offset = len(all_doc_ids)
        df = len(pl.doc_ids)
        term_meta[t] = {"tid": tid, "df": df, "offset": offset}

        all_doc_ids.extend(pl.doc_ids)
        all_tfs.extend(pl.tfs)
        for positions in pl.positions:
            all_positions.extend(positions)
            pos_offsets.append(len(all_positions))

    np.savez_compressed(
        base / "postings.npz",
        doc_ids=np.asarray(all_doc_ids, dtype=np.int32),
        tfs=np.asarray(all_tfs, dtype=np.int16),
        positions=np.asarray(all_positions, dtype=np.int32),
        pos_offsets=np.asarray(pos_offsets, dtype=np.int64),   # 长度 = 总 posting 数 + 1
    )

    with open(base / "term_dict.json", "w", encoding="utf-8") as f:
        json.dump(term_meta, f, ensure_ascii=False)

    with open(base / "doc_map.json", "w", encoding="utf-8") as f:
        json.dump({str(v): k for k, v in nid_to_int.items()}, f, ensure_ascii=False)

def main():
    long_sorted,nid_to_int=read_preprocess(BASE)
    word_idx=build_idx(long_sorted)
    save_to_disk(BASE,word_idx,nid_to_int)

if __name__=="__main__":
    main()
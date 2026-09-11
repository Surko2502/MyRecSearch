import pandas as pd
import json
import numpy as np
from pathlib import Path
BASE = Path(r"D:\RecSeach\Search")

def read_preprocess(path:Path): 
    df=pd.read_parquet(path)
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

def build_idx(long_sorted:pd.DataFrame):
    word_idx={} # term -> [offset,df]
    all_doc_ids=[]
    all_position=[]
    pos_offset=[]
    tfs=[]

    cur_term=None
    cur_doc_id=None
    doc_cnt=0
    pos_cnt=0
    for term, doc_id, pos in zip(
        long_sorted["term"], long_sorted["doc_id"], long_sorted["pos"]
    ):
        if term!=cur_term:
            pos_offset.append(pos_cnt)
            all_doc_ids.append(doc_id)
            word_idx[term]=[doc_cnt,1]

            cur_term=term
            cur_doc_id=doc_id

            doc_cnt+=1
            pos_cnt+=1
            all_position.append(pos)

        elif doc_id!=cur_doc_id:
            pos_offset.append(pos_cnt)
            all_doc_ids.append(doc_id)
            word_idx[term][1]+=1

            cur_doc_id=doc_id

            doc_cnt+=1
            pos_cnt+=1
            all_position.append(pos)

        else:
            pos_cnt+=1
            all_position.append(pos)
    for i,t in enumerate(pos_offset):
        if i+1<=doc_cnt-1:
            tfs.append(pos_offset[i+1]-t)
    tfs.append(pos_cnt-pos_offset[-1])
    pos_offset.append(pos_cnt)
    return word_idx,all_doc_ids,all_position,pos_offset,tfs
        
# 写入磁盘
def save_dict_to_json(path:Path,nid_to_int:dict):
    with open(path / "doc_map.json", "w", encoding="utf-8") as f:
        json.dump({str(v): k for k, v in nid_to_int.items()}, f, ensure_ascii=False)

def save_idx_to_disk_old(path:Path,word_idx:dict):
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
        path / "postings.npz",
        doc_ids=np.asarray(all_doc_ids, dtype=np.int32),
        tfs=np.asarray(all_tfs, dtype=np.int16),
        positions=np.asarray(all_positions, dtype=np.int32),
        pos_offsets=np.asarray(pos_offsets, dtype=np.int64),   # 长度 = 总 posting 数 + 1
    )

    with open(path / "term_dict.json", "w", encoding="utf-8") as f:
        json.dump(term_meta, f, ensure_ascii=False)

def save_idx_to_disk(path:Path,word_idx:dict,all_doc_ids:list[int],all_positions:list[int],pos_offsets:list[int],tfs:list[int]):
    terms = sorted(word_idx.keys())
    term_meta = {}   
    for tid, t in enumerate(terms):
        term_meta[t] = {"tid": tid, "df": word_idx.get(t)[1], "offset": word_idx.get(t)[0]}
    np.savez_compressed(
            path / "postings.npz",
            doc_ids=np.asarray(all_doc_ids, dtype=np.int32),
            tfs=np.asarray(tfs, dtype=np.int16),
            positions=np.asarray(all_positions, dtype=np.int32),
            pos_offsets=np.asarray(pos_offsets, dtype=np.int64),  
        )
    
    with open(path / "term_dict.json", "w", encoding="utf-8") as f:
        json.dump(term_meta, f, ensure_ascii=False)
def main():
    long_sorted,nid_to_int=read_preprocess(BASE  / "tokens.parquet")
    word_idx,all_doc_ids,all_position,pos_offset,tfs=build_idx(long_sorted)
    save_idx_to_disk(BASE,word_idx,all_doc_ids,all_position,pos_offset,tfs)
    # word_idx=build_idx(long_sorted)
    # save_idx_to_disk(BASE,word_idx)
    save_dict_to_json(BASE,nid_to_int)

if __name__=="__main__":
    main()
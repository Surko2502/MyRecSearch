import json
import numpy as np
from config import *
class Searcher:
    def __init__(self, base: Path = BASE):
        # 词典：term -> {tid, df, offset}
        with open(base / "term_dict.json", "r", encoding="utf-8") as f:
            self.term_meta = json.load(f)

        # 倒排表：npz，懒加载
        data = np.load(base / "postings.npz")
        self.all_doc_ids = data["doc_ids"]
        self.all_tfs = data["tfs"]
        self.all_positions = data["positions"]
        self.pos_offsets = data["pos_offsets"]

        # 文档映射：JSON key 是字符串，转回 int
        with open(base / "doc_map.json", "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.doc_id_to_news_id = {int(k): v for k, v in raw.items()}

    def search(self, term: str, topk: int | None = None):
        """
        单词检索。返回 list[dict]：
            {"doc_id": int, "news_id": str, "tf": int, "positions": list[int]}
        未命中返回 []。
        """
        meta = self.term_meta.get(term)
        if meta is None:
            return []

        offset = meta["offset"]
        df = meta["df"]

        doc_ids = self.all_doc_ids[offset: offset + df]
        tfs = self.all_tfs[offset: offset + df]

        results = []
        for i in range(df):
            gid = offset + i                      # 全局 posting 下标
            s = self.pos_offsets[gid]
            e = self.pos_offsets[gid + 1]
            positions = self.all_positions[s:e].tolist()
            did = int(doc_ids[i])
            results.append({
                "doc_id": did,
                "news_id": self.doc_id_to_news_id.get(did),
                "tf": int(tfs[i]),
                "positions": positions,
            })

        if topk is not None:
            results.sort(key=lambda r: r["tf"], reverse=True)
            results = results[:topk]

        return results

    def stats(self, term: str):
        """返回该词的 df（命中文档数）和 cf（总词频）。"""
        meta = self.term_meta.get(term)
        if meta is None:
            return {"term": term, "df": 0, "cf": 0}
        offset, df = meta["offset"], meta["df"]
        cf = int(self.all_tfs[offset: offset + df].sum())
        return {"term": term, "df": df, "cf": cf}
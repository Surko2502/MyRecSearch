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

    def _postings(self, term: str):
        """取词的倒排表切片 (doc_ids, tfs)，二者按 doc_id 升序对齐；
        词不存在返回 None。"""
        meta = self.term_meta.get(term)
        if meta is None:
            return None
        offset = meta["offset"]
        df = meta["df"]
        return (
            self.all_doc_ids[offset: offset + df],
            self.all_tfs[offset: offset + df],
        )

    @staticmethod
    def _and_merge(a_ids, a_tf, b_ids, b_tf):
        """两个升序倒排表的 AND 合并（双指针 merge 法）。

        返回 (交集 doc_ids, 对应 tf 之和)。a_tf / b_tf 与各自 doc_ids 对齐。
        """
        out_ids = []
        out_tf = []
        i = j = 0
        na, nb = len(a_ids), len(b_ids)
        while i < na and j < nb:
            ai, bj = a_ids[i], b_ids[j]
            if ai == bj:
                out_ids.append(ai)
                out_tf.append(a_tf[i] + b_tf[j])
                i += 1
                j += 1
            elif ai < bj:
                i += 1
            else:
                j += 1
        return out_ids, out_tf

    def search_terms(self, terms: list[str], topk: int | None = None):
        """多词 AND 检索（merge 法）：返回同时包含所有词的文档。

        terms: 词列表；任一词未命中返回 []（AND 语义）。
        topk: 按 tf 之和降序取前 k 篇。
        返回 list[dict]：{"doc_id": int, "news_id": str, "tf": int}。
        """
        if not terms:
            return []
        terms = list(dict.fromkeys(terms))       # 去重，避免同一词重复计分

        postings = []
        for t in terms:
            p = self._postings(t)
            if p is None:
                return []                        # 有一个词不存在 -> 交集为空
            postings.append(p)

        # 先合并 df 最小的列表，减少指针移动次数
        postings.sort(key=lambda p: len(p[0]))

        cur_ids = postings[0][0].tolist()
        cur_tf = postings[0][1].tolist()
        for ids, tfs in postings[1:]:
            cur_ids, cur_tf = self._and_merge(
                cur_ids, cur_tf, ids.tolist(), tfs.tolist()
            )
            if not cur_ids:
                return []

        results = [
            {
                "doc_id": did,
                "news_id": self.doc_id_to_news_id.get(did),
                "tf": tf,
            }
            for did, tf in zip(cur_ids, cur_tf)
        ]

        if topk is not None:
            results.sort(key=lambda r: r["tf"], reverse=True)
            results = results[:topk]

        return results
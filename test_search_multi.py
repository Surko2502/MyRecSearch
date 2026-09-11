from search import Searcher


def main():
    s = Searcher()

    print(f"词典大小: {len(s.term_meta)}")
    print(f"总 doc 数: {len(s.doc_id_to_news_id)}")

    # 多词 AND 检索测试（merge 法）
    queries = [
        ["the", "apple"],                       # 普通两词
        ["the", "apple", "new"],                # 三词，验证 k 路 merge
        ["apple", "__not_a_real_term__"],       # 含不存在词 -> 应为空
        [],                                     # 空列表 -> 应为空
    ]

    for terms in queries:
        res = s.search_terms(terms)
        print(f"\n=== 多词查询: {terms} ===")
        if not res:
            print("  (无命中)")
            continue

        # 交叉验证：命中集合应与逐个单词检索的交集一致
        inter = {r["doc_id"] for r in s.search(terms[0])}
        for t in terms[1:]:
            inter &= {r["doc_id"] for r in s.search(t)}
        ok = {r["doc_id"] for r in res} == inter
        print(f"  命中 {len(res)} 篇，交叉验证: {'通过' if ok else '失败'}")

        for r in res[:5]:
            print(f"  news_id={r['news_id']}  doc_id={r['doc_id']}  tf={r['tf']}")
        if len(res) > 5:
            print(f"  ... 共 {len(res)} 篇")

    # topk 测试：按 tf 之和降序取前 k 篇
    top = s.search_terms(["the", "apple"], topk=3)
    print(f"\n=== topk=3: ['the', 'apple'] -> {len(top)} 篇 ===")
    for r in top:
        print(f"  news_id={r['news_id']}  doc_id={r['doc_id']}  tf={r['tf']}")


if __name__ == "__main__":
    main()

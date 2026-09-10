from search import Searcher


def main():
    s = Searcher()

    print(f"词典大小: {len(s.term_meta)}")
    print(f"总 doc 数: {len(s.doc_id_to_news_id)}")
    print(f"总 posting 数: {len(s.all_doc_ids)}")

    # 手动指定要测的词
    queries = ["中国", "经济", "the", "apple", "__not_a_real_term__"]

    for q in queries:
        res = s.search(q)
        st = s.stats(q)
        print(f"\n=== 查询: {q!r} ===")
        print(f"df={st['df']}, cf={st['cf']}")
        if not res:
            print("  (无命中)")
            continue
        for r in res[:5]:
            print(f"  news_id={r['news_id']}  doc_id={r['doc_id']}  "
                  f"tf={r['tf']}  positions={r['positions']}")
        if len(res) > 5:
            print(f"  ... 共 {len(res)} 篇")


if __name__ == "__main__":
    main()
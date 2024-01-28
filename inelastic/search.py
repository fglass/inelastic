import time

from inelastic.index import index
from inelastic.query import Result, query

# https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-abstract1.xml.gz
DATA_DUMP = "./enwiki-latest-abstract1.xml"
PAGE_SIZE = 10


def search():
    t1 = time.time()
    docs = _load()
    print(f"Loaded {len(docs)} documents in {time.time() - t1:.2f}s")

    t2 = time.time()
    idx = index(docs)
    print(f"Indexed {len(idx.inverted)} terms in {time.time() - t2:.2f}s")

    while True:
        q = input("Search: ")

        tq = time.time()
        results = query(idx, q)
        duration_ms = (time.time() - tq) * 1000

        _pprint(results, docs, duration_ms)


def _load() -> list[tuple[str, str]]:
    docs = []

    with open(DATA_DUMP) as f:
        doc_title = ""
        for line in f.readlines():
            if line.startswith("<title>"):
                doc_title = line.removeprefix("<title>Wikipedia: ").removesuffix(
                    "</title>\n"
                )
            if line.startswith("<abstract>"):
                doc_content = line.removeprefix("<abstract>").removesuffix(
                    "</abstract>\n"
                )
                docs.append((doc_title, doc_content))
                doc_title = ""

    return docs


def _pprint(results: list[Result], docs: list[tuple[str, str]], duration_ms: float):
    print(f"Found {len(results)} results in {duration_ms:.2f}ms")

    if len(results) == 0:
        return

    print(f"Top {min(len(results), PAGE_SIZE)} results:")
    results = results[:PAGE_SIZE]

    for i, r in enumerate(results):
        title, content = docs[r.doc_id]
        print(f"\t{i + 1}. [{r.score:.2f}] {r.doc_id} {title}: {content[:50]}...")


if __name__ == "__main__":
    search()

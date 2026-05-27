import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan


DEFAULT_ES_URL = "http://127.0.0.1:9200"
DEFAULT_INDEX = "scholarmind_knowledge_base"


def _sort_key(chunk: dict[str, Any]) -> tuple[int, str]:
    metadata = chunk.get("metadata") or {}
    global_index = metadata.get("global_chunk_index")
    if not isinstance(global_index, int):
        global_index = 10**12
    return global_index, str(metadata.get("chunk_id") or chunk.get("_id") or "")


def format_chunk(chunk: dict[str, Any], ordinal: int) -> str:
    metadata = chunk.get("metadata") or {}
    text = str(chunk.get("text") or "").strip()
    return "\n".join(
        [
            "=" * 100,
            f"CHUNK {ordinal}",
            f"ES_ID: {chunk.get('_id', '')}",
            f"SECTION: {metadata.get('section', '')}",
            f"SUBSECTION: {metadata.get('subsection', '')}",
            f"CHUNK_TYPE: {metadata.get('chunk_type', '')}",
            f"CHUNK_ID: {metadata.get('chunk_id', '')}",
            f"PARENT_CHUNK_ID: {metadata.get('parent_chunk_id', '')}",
            "METADATA:",
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
            "TEXT:",
            text,
            "",
        ]
    )


def read_chunks(es: Elasticsearch, index: str, batch_size: int = 500) -> list[dict[str, Any]]:
    docs = []
    for hit in scan(
        es,
        index=index,
        query={"query": {"match_all": {}}},
        size=batch_size,
        _source=["text", "metadata"],
    ):
        source = hit.get("_source") or {}
        docs.append(
            {
                "_id": hit.get("_id", ""),
                "text": source.get("text", ""),
                "metadata": source.get("metadata") or {},
            }
        )
    return sorted(docs, key=_sort_key)


def write_chunks(chunks: Iterable[dict[str, Any]], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as file:
        for count, chunk in enumerate(chunks, start=1):
            file.write(format_chunk(chunk, count))
            file.write("\n")
    return count


def export_chunks(es_url: str, index: str, output_path: Path, batch_size: int = 500) -> int:
    es = Elasticsearch(es_url)
    chunks = read_chunks(es, index=index, batch_size=batch_size)
    return write_chunks(chunks, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export indexed ScholarMind chunks with metadata to a text file.")
    parser.add_argument("--es-url", default=DEFAULT_ES_URL)
    parser.add_argument("--index", default=DEFAULT_INDEX)
    parser.add_argument("--output", default=str(Path("..") / "exports" / "scholarmind_chunks.txt"))
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()

    output_path = Path(args.output).resolve()
    count = export_chunks(
        es_url=args.es_url,
        index=args.index,
        output_path=output_path,
        batch_size=args.batch_size,
    )
    print(f"Exported {count} chunks to {output_path}")


if __name__ == "__main__":
    main()

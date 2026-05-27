from dataclasses import dataclass, field


@dataclass
class RetrievalHit:
    chunk_id: str
    content: str
    source: str
    page: int | str
    section: str
    score: float
    retriever: str
    metadata: dict = field(default_factory=dict)

    @property
    def identity(self) -> str:
        return self.chunk_id or f"{self.source}:{self.page}:{self.section}:{hash(self.content)}"

    @property
    def context_text(self) -> str:
        return str(self.metadata.get("parent_text") or self.content)

    def to_metadata(self) -> dict:
        merged = dict(self.metadata)
        merged.update(
            {
                "chunk_id": self.chunk_id,
                "source": self.source,
                "page": self.page,
                "section": self.section,
                "score": self.score,
                "retriever": self.retriever,
            }
        )
        return merged

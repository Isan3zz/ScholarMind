import unittest

from app.rag.paper_chunker import build_paper_chunks
from app.rag.paper_sections import PaperTextUnit


def _children(chunks):
    """返回所有子块（chunk_type != 'parent' 且 != 'paper_metadata'）。"""
    return [c for c in chunks if c.metadata.get("chunk_type") not in ("parent", "paper_metadata")]


def _parents(chunks):
    """返回所有父块（chunk_type == 'parent'）。"""
    return [c for c in chunks if c.metadata.get("chunk_type") == "parent"]


def _parent_by_id(chunks, pid):
    for c in chunks:
        if c.metadata.get("chunk_id") == pid:
            return c
    return None


class PaperChunkerTest(unittest.TestCase):
    def test_build_paper_chunks_keeps_section_metadata_and_parent_ids(self):
        units = [
            PaperTextUnit(text="This is the abstract.", section="Abstract", chunk_type="abstract"),
            PaperTextUnit(text="We propose a retriever.", section="Method", chunk_type="body"),
        ]

        chunks = build_paper_chunks(units, source="paper.pdf", paper_title="IRIS")

        children = _children(chunks)
        self.assertEqual(len(children), 2)
        self.assertEqual(children[0].metadata["section"], "Abstract")
        self.assertEqual(children[0].metadata["source"], "paper.pdf")
        self.assertEqual(children[0].metadata["paper_title"], "IRIS")
        self.assertIn("parent_chunk_id", children[0].metadata)
        self.assertIn("chunk_id", children[0].metadata)
        # 子块不应再有 parent_text
        self.assertNotIn("parent_text", children[0].metadata)
        # 应有独立父块
        self.assertTrue(len(_parents(chunks)) >= 2)

    def test_build_paper_chunks_preserves_authors_metadata(self):
        units = [
            PaperTextUnit(text="This is the abstract.", section="Abstract", chunk_type="abstract"),
        ]

        chunks = build_paper_chunks(
            units,
            source="paper.pdf",
            paper_title="IRIS",
            authors=["Jane Doe", "杭 雨聪"],
        )

        self.assertEqual(_children(chunks)[0].metadata["authors"], ["Jane Doe", "杭 雨聪"])

    def test_build_paper_chunks_adds_single_paper_metadata_chunk_for_title_and_authors(self):
        units = [
            PaperTextUnit(text="This is the abstract.", section="Abstract", chunk_type="abstract"),
            PaperTextUnit(text="Method paragraph.", section="Method"),
        ]

        chunks = build_paper_chunks(
            units,
            source="paper.pdf",
            paper_title="IRIS",
            authors=["Jane Doe", "杭 雨聪"],
        )

        metadata_chunk = chunks[0]
        self.assertEqual(metadata_chunk.metadata["chunk_type"], "paper_metadata")
        self.assertEqual(metadata_chunk.metadata["section"], "Metadata")
        self.assertEqual(metadata_chunk.metadata["global_chunk_index"], 0)
        self.assertIn("Paper Title: IRIS", metadata_chunk.page_content)
        self.assertIn("Authors: Jane Doe, 杭 雨聪", metadata_chunk.page_content)
        self.assertEqual(_children(chunks)[0].metadata["global_chunk_index"], 1)

    def test_subsection_children_share_parent_with_all_siblings(self):
        units = [
            PaperTextUnit(text="Intro paragraph.", section="Introduction"),
            PaperTextUnit(text="Architecture paragraph 1.", section="Method", subsection="Architecture"),
            PaperTextUnit(text="Architecture paragraph 2.", section="Method", subsection="Architecture"),
            PaperTextUnit(text="Objective paragraph.", section="Method", subsection="Objective"),
            PaperTextUnit(text="[1] Reference", section="References", chunk_type="reference"),
        ]

        chunks = build_paper_chunks(units, source="paper.pdf", paper_title="IRIS")

        children = _children(chunks)
        self.assertEqual(len(children), 4)  # Introduction + Architecture×2 + Objective
        self.assertEqual(children[0].metadata["section_id"], "introduction")
        self.assertEqual(children[1].metadata["subsection_id"], "method::architecture")
        self.assertEqual(children[1].metadata["global_chunk_index"], 2)
        self.assertIn("[IRIS | Method | Architecture]", children[1].page_content)

        # 子块不应有 parent_text
        self.assertNotIn("parent_text", children[1].metadata)

        # 父块：Architecture 的两个子块共享同一个父块
        arch_parents = [p for p in _parents(chunks) if p.metadata["subsection_id"] == "method::architecture"]
        self.assertEqual(len(arch_parents), 1)
        self.assertIn("Architecture paragraph 1.", arch_parents[0].page_content)
        self.assertIn("Architecture paragraph 2.", arch_parents[0].page_content)
        self.assertNotIn("Objective paragraph.", arch_parents[0].page_content)

        # 父块的 chunk_id 应与子块的 parent_chunk_id 对应
        self.assertEqual(children[1].metadata["parent_chunk_id"], arch_parents[0].metadata["chunk_id"])

    def test_non_subsection_parent_includes_neighbors(self):
        units = [
            PaperTextUnit(text="Paragraph 0.", section="Introduction"),
            PaperTextUnit(text="Paragraph 1.", section="Introduction"),
            PaperTextUnit(text="Paragraph 2.", section="Introduction"),
        ]

        chunks = build_paper_chunks(units, source="paper.pdf", paper_title="IRIS")

        children = _children(chunks)
        # 中间子块的父块应包含前后邻居
        mid_child = children[1]  # Paragraph 1
        mid_parent = _parent_by_id(chunks, mid_child.metadata["parent_chunk_id"])
        self.assertIsNotNone(mid_parent)
        self.assertIn("Paragraph 0.", mid_parent.page_content)
        self.assertIn("Paragraph 1.", mid_parent.page_content)
        self.assertIn("Paragraph 2.", mid_parent.page_content)


if __name__ == "__main__":
    unittest.main()

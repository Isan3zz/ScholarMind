import unittest

from app.rag.tei_parser import parse_tei_to_paper


TEI_SAMPLE = """<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt><title>IRIS Paper</title></titleStmt>
      <sourceDesc>
        <biblStruct>
          <analytic>
            <author>
              <persName>
                <forename>Jane</forename>
                <surname>Doe</surname>
              </persName>
            </author>
            <author>
              <persName>
                <forename>杭</forename>
                <surname>雨聪</surname>
              </persName>
            </author>
          </analytic>
        </biblStruct>
      </sourceDesc>
    </fileDesc>
    <profileDesc>
      <abstract><p>This is the abstract.</p></abstract>
    </profileDesc>
  </teiHeader>
  <text>
    <body>
      <div>
        <head>1 Introduction</head>
        <p>Intro paragraph.</p>
        <div>
          <head>Motivation</head>
          <p>Motivation paragraph.</p>
        </div>
      </div>
      <div>
        <head>2 Method</head>
        <p>Method paragraph.</p>
      </div>
    </body>
  </text>
</TEI>"""


class TeiParserTest(unittest.TestCase):
    def test_parse_tei_to_paper_extracts_title_sections_and_subsections(self):
        paper = parse_tei_to_paper(TEI_SAMPLE)

        self.assertEqual(paper.title, "IRIS Paper")
        self.assertEqual(paper.authors, ["Jane Doe", "杭 雨聪"])
        self.assertEqual(paper.units[0].section, "Abstract")
        self.assertEqual(paper.units[0].chunk_type, "abstract")
        self.assertEqual(paper.units[1].section, "Introduction")
        self.assertEqual(paper.units[1].subsection, "")
        self.assertEqual(paper.units[2].section, "Introduction")
        self.assertEqual(paper.units[2].subsection, "Motivation")
        self.assertEqual(paper.units[3].section, "Method")

    def test_parse_tei_to_paper_uses_head_number_for_flat_subsections(self):
        tei = """<TEI xmlns="http://www.tei-c.org/ns/1.0">
          <teiHeader>
            <fileDesc>
              <titleStmt><title>Numbered Paper</title></titleStmt>
              <sourceDesc><biblStruct><analytic /></biblStruct></sourceDesc>
            </fileDesc>
          </teiHeader>
          <text>
            <body>
              <div>
                <head n="5">Experiments</head>
                <p>Experiment overview.</p>
              </div>
              <div>
                <head n="5.1">Experimental Setup</head>
                <p>Setup details.</p>
              </div>
              <div>
                <head n="5.2">Experimental Results</head>
                <p>Result details.</p>
              </div>
              <div>
                <head n="6">Limitations</head>
                <p>Limitations details.</p>
              </div>
            </body>
          </text>
        </TEI>"""

        paper = parse_tei_to_paper(tei)

        self.assertEqual(paper.units[0].section, "Experiments")
        self.assertEqual(paper.units[0].subsection, "")
        self.assertEqual(paper.units[1].section, "Experiments")
        self.assertEqual(paper.units[1].subsection, "Experimental Setup")
        self.assertEqual(paper.units[2].section, "Experiments")
        self.assertEqual(paper.units[2].subsection, "Experimental Results")
        self.assertEqual(paper.units[3].section, "Limitations")
        self.assertEqual(paper.units[3].subsection, "")

    def test_parse_tei_to_paper_keeps_numbered_heading_as_section_when_parent_missing(self):
        tei = """<TEI xmlns="http://www.tei-c.org/ns/1.0">
          <teiHeader>
            <fileDesc>
              <titleStmt><title>Missing Parent Paper</title></titleStmt>
              <sourceDesc><biblStruct><analytic /></biblStruct></sourceDesc>
            </fileDesc>
          </teiHeader>
          <text>
            <body>
              <div>
                <head n="4.4">Inference Phase: Construct New Token Distribution</head>
                <p>Inference details.</p>
              </div>
            </body>
          </text>
        </TEI>"""

        paper = parse_tei_to_paper(tei)

        self.assertEqual(paper.units[0].section, "Inference Phase: Construct New Token Distribution")
        self.assertEqual(paper.units[0].subsection, "")

    def test_parse_tei_to_paper_detects_appendix_number_from_heading_text(self):
        tei = """<TEI xmlns="http://www.tei-c.org/ns/1.0">
          <teiHeader>
            <fileDesc>
              <titleStmt><title>Appendix Paper</title></titleStmt>
              <sourceDesc><biblStruct><analytic /></biblStruct></sourceDesc>
            </fileDesc>
          </teiHeader>
          <text>
            <body>
              <div>
                <head>C.1 SafeDecoding is Safe</head>
                <p>Appendix safety details.</p>
              </div>
            </body>
          </text>
        </TEI>"""

        paper = parse_tei_to_paper(tei)

        self.assertEqual(paper.units[0].section, "C.1 SafeDecoding is Safe")
        self.assertEqual(paper.units[0].subsection, "")

    def test_parse_tei_to_paper_does_not_attach_appendix_number_to_previous_section(self):
        tei = """<TEI xmlns="http://www.tei-c.org/ns/1.0">
          <teiHeader>
            <fileDesc>
              <titleStmt><title>Appendix Paper</title></titleStmt>
              <sourceDesc><biblStruct><analytic /></biblStruct></sourceDesc>
            </fileDesc>
          </teiHeader>
          <text>
            <body>
              <div>
                <head n="6">Limitations</head>
                <p>Limitations details.</p>
              </div>
              <div>
                <head>C.1 SafeDecoding is Safe</head>
                <p>Appendix safety details.</p>
              </div>
            </body>
          </text>
        </TEI>"""

        paper = parse_tei_to_paper(tei)

        self.assertEqual(paper.units[1].section, "C.1 SafeDecoding is Safe")
        self.assertEqual(paper.units[1].subsection, "")

    def test_parse_tei_to_paper_uses_appendix_parent_from_heading_text(self):
        tei = """<TEI xmlns="http://www.tei-c.org/ns/1.0">
          <teiHeader>
            <fileDesc>
              <titleStmt><title>Appendix Paper</title></titleStmt>
              <sourceDesc><biblStruct><analytic /></biblStruct></sourceDesc>
            </fileDesc>
          </teiHeader>
          <text>
            <body>
              <div>
                <head>C Additional Results</head>
                <p>Appendix overview.</p>
              </div>
              <div>
                <head>C.1 SafeDecoding is Safe</head>
                <p>Appendix safety details.</p>
              </div>
            </body>
          </text>
        </TEI>"""

        paper = parse_tei_to_paper(tei)

        self.assertEqual(paper.units[0].section, "C Additional Results")
        self.assertEqual(paper.units[0].subsection, "")
        self.assertEqual(paper.units[1].section, "C Additional Results")
        self.assertEqual(paper.units[1].subsection, "C.1 SafeDecoding is Safe")

    def test_parse_tei_to_paper_merges_formula_with_neighboring_paragraphs(self):
        tei = """<TEI xmlns="http://www.tei-c.org/ns/1.0">
          <teiHeader>
            <fileDesc>
              <titleStmt><title>Formula Paper</title></titleStmt>
              <sourceDesc><biblStruct><analytic /></biblStruct></sourceDesc>
            </fileDesc>
          </teiHeader>
          <text>
            <body>
              <div>
                <head>2 Method</head>
                <p>SafeDecoding constructs a sample space represented as:</p>
                <formula xml:id="formula_4">V(c)_n = arg min S s.t. |S| &gt;= c.</formula>
                <p>Here V tokens represent the top candidates from both models.</p>
              </div>
            </body>
          </text>
        </TEI>"""

        paper = parse_tei_to_paper(tei)

        self.assertEqual(len(paper.units), 1)
        self.assertEqual(paper.units[0].section, "Method")
        self.assertIn("SafeDecoding constructs a sample space", paper.units[0].text)
        self.assertIn("[Formula formula_4] V(c)_n = arg min S s.t. |S| >= c.", paper.units[0].text)
        self.assertIn("Here V tokens represent the top candidates", paper.units[0].text)

    def test_parse_tei_to_paper_removes_inline_figure_caption_but_keeps_footnote(self):
        tei = """<TEI xmlns="http://www.tei-c.org/ns/1.0">
          <teiHeader>
            <fileDesc>
              <titleStmt><title>Caption Paper</title></titleStmt>
              <sourceDesc><biblStruct><analytic /></biblStruct></sourceDesc>
            </fileDesc>
          </teiHeader>
          <text>
            <body>
              <div>
                <head>1 Introduction</head>
                <p>Reports of LLMs producing biased content. 1 Our code is publicly available at: https://github.com/uw-nsl/SafeDecoding. The words in red are GCG suffixes. We note that although the token representing the word "Sure" has a dominant probability, safety disclaimers such as "I", "Sorry", and "As" are still present in the sample space, which is sorted in descending order in token probabilities. When a safety disclaimer token is sampled, the model would reject the attacker's harmful query. Robust safety measures are needed.</p>
              </div>
            </body>
          </text>
        </TEI>"""

        paper = parse_tei_to_paper(tei)
        text = paper.units[0].text

        self.assertIn("Our code is publicly available", text)
        self.assertIn("https://github.com/uw-nsl/SafeDecoding", text)
        self.assertIn("Robust safety measures are needed.", text)
        self.assertNotIn("The words in red are GCG suffixes", text)
        self.assertNotIn("safety disclaimer token is sampled", text)

    def test_parse_tei_to_paper_skips_figure_elements(self):
        tei = """<TEI xmlns="http://www.tei-c.org/ns/1.0">
          <teiHeader>
            <fileDesc>
              <titleStmt><title>Figure Paper</title></titleStmt>
              <sourceDesc><biblStruct><analytic /></biblStruct></sourceDesc>
            </fileDesc>
          </teiHeader>
          <text>
            <body>
              <div>
                <head>1 Introduction</head>
                <p>Main body evidence.</p>
                <figure>
                  <head>Figure 1</head>
                  <figDesc>This caption should not be indexed as body text.</figDesc>
                </figure>
                <p>More body evidence.</p>
              </div>
            </body>
          </text>
        </TEI>"""

        paper = parse_tei_to_paper(tei)
        combined = "\n".join(unit.text for unit in paper.units)

        self.assertIn("Main body evidence.", combined)
        self.assertIn("More body evidence.", combined)
        self.assertNotIn("This caption should not be indexed", combined)


if __name__ == "__main__":
    unittest.main()

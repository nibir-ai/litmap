"""Tests for litmap.arxiv — HTTP calls are mocked."""
from unittest.mock import MagicMock, patch
import pytest
from litmap.arxiv import Paper, _parse_feed, fetch_papers, fetch_by_id

SAMPLE_FEED = '''<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2310.00001v1</id>
    <title>Attention Is All You Need Redux</title>
    <summary>We revisit the transformer architecture for language tasks.</summary>
    <published>2023-10-01T00:00:00Z</published>
    <updated>2023-10-05T00:00:00Z</updated>
    <author><name>Alice Smith</name></author>
    <author><name>Bob Jones</name></author>
    <category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>
    <arxiv:doi>10.1234/test.001</arxiv:doi>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2310.00002v1</id>
    <title>Graph Neural Networks for Chemistry</title>
    <summary>GNNs for molecular property prediction in drug discovery.</summary>
    <published>2023-10-02T00:00:00Z</published>
    <updated>2023-10-02T00:00:00Z</updated>
    <author><name>Carol Lee</name></author>
    <category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
</feed>'''

EMPTY_FEED = '''<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
</feed>'''


def test_parse_count():
    assert len(_parse_feed(SAMPLE_FEED)) == 2

def test_parse_id():
    assert "2310.00001" in _parse_feed(SAMPLE_FEED)[0].arxiv_id

def test_parse_title():
    assert _parse_feed(SAMPLE_FEED)[0].title == "Attention Is All You Need Redux"

def test_parse_authors():
    assert _parse_feed(SAMPLE_FEED)[0].authors == ["Alice Smith", "Bob Jones"]

def test_parse_categories():
    p = _parse_feed(SAMPLE_FEED)[0]
    assert "cs.LG" in p.categories and "cs.CL" in p.categories

def test_parse_published():
    assert _parse_feed(SAMPLE_FEED)[0].published == "2023-10-01"

def test_parse_doi():
    assert _parse_feed(SAMPLE_FEED)[0].doi == "10.1234/test.001"

def test_parse_no_doi():
    assert _parse_feed(SAMPLE_FEED)[1].doi is None

def test_parse_empty():
    assert _parse_feed(EMPTY_FEED) == []

def _mock_resp(text):
    m = MagicMock()
    m.text = text
    m.raise_for_status = MagicMock()
    return m

@patch("litmap.arxiv.requests.get")
@patch("litmap.arxiv.time.sleep")
def test_fetch_papers_returns_papers(mock_sleep, mock_get):
    mock_get.return_value = _mock_resp(SAMPLE_FEED)
    papers = fetch_papers("transformer", max_results=2)
    assert len(papers) == 2
    assert all(isinstance(p, Paper) for p in papers)

@patch("litmap.arxiv.requests.get")
@patch("litmap.arxiv.time.sleep")
def test_fetch_empty(mock_sleep, mock_get):
    mock_get.return_value = _mock_resp(EMPTY_FEED)
    assert fetch_papers("test") == []

@patch("litmap.arxiv.requests.get")
@patch("litmap.arxiv.time.sleep")
def test_fetch_by_id_found(mock_sleep, mock_get):
    mock_get.return_value = _mock_resp(SAMPLE_FEED)
    assert isinstance(fetch_by_id("2310.00001"), Paper)

@patch("litmap.arxiv.requests.get")
@patch("litmap.arxiv.time.sleep")
def test_fetch_by_id_not_found(mock_sleep, mock_get):
    mock_get.return_value = _mock_resp(EMPTY_FEED)
    assert fetch_by_id("9999.99999") is None

def test_max_results_guard():
    with pytest.raises(ValueError):
        fetch_papers("test", max_results=101)

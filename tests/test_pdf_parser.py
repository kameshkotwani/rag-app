"""
Test for the PDF parser module.
"""
import os
import pytest
from src.utils.pdf_parser import load_and_split

def test_load_and_split():
    """
    Test that the load_and_split function returns non-zero chunks for a sample PDF.
    
    This test requires a sample PDF file to be available.
    If no sample PDF is available, the test will be skipped.
    """
    # Path to a sample PDF file for testing
    # This sample file should be available in your test data directory
    sample_pdf_path = os.path.join(os.path.dirname(__file__), "data", "sample.pdf")
    
    # Skip test if sample PDF not available
    if not os.path.exists(sample_pdf_path):
        pytest.skip("Sample PDF not available for testing")
    
    # Load and split the PDF
    docs = load_and_split(sample_pdf_path)
    
    # Check that docs is a non-empty list
    assert docs, "No documents returned from load_and_split"
    assert len(docs) > 0, "Empty list of documents returned from load_and_split"
    
    # Check that each document has content
    for doc in docs:
        assert doc.page_content, "Document has empty page_content"
        assert isinstance(doc.page_content, str), "Document page_content is not a string"
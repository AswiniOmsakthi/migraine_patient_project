# -*- coding: utf-8 -*-
import json
import pytest
from langchain.agents import Tool
from src.knowledge_stores.migraine import load_vectorstore, migraine_country_brand_data

@pytest.fixture(scope="module")
def tools():
    return load_vectorstore(None)

def test_vectorstore_tools_loaded(tools):
    names = {t.name for t in tools}
    expected = {
        "migraine_qa_csv",
        "migraine_country_brand_usage",
        "pubmed_search",
        "clinical_trials_search"
    }
    missing = expected - names
    assert not missing, f"Missing tools: {sorted(missing)}"
    assert len(tools) >= len(expected) + 1, f"Expected ≥ {len(expected)+1} tools, got {len(tools)}"
    assert all(isinstance(t, Tool) for t in tools)

def test_country_brand_data_json():
    raw = migraine_country_brand_data("")
    arr = json.loads(raw)
    assert isinstance(arr, list), "Expected list"
    assert arr, "List must not be empty"
    rec = arr[0]
    for key in ["Country", "Brand Name", "Chronic Migraine"]:
        assert key in rec, f"Missing key: {key}"

def test_qa_csv_tool_output(tools):
    qa = next(t for t in tools if t.name == "migraine_qa_csv")
    out = qa.func("migraine")
    assert isinstance(out, str)
    assert out.strip(), "Empty output from CSV tool"

def test_pdf_tool_output_minimal(tools):
    core = {"migraine_qa_csv", "migraine_country_brand_usage", "pubmed_search", "clinical_trials_search"}
    pdfs = [t for t in tools if t.name not in core]
    assert pdfs, "No PDF-based tools found"
    sample = pdfs[0]
    out = sample.func("headache prevention")
    assert isinstance(out, str)
    assert out.strip(), f"Empty output from PDF tool {sample.name}"

@pytest.mark.network
def test_pubmed_search_tool_lives(tools):
    pub = next(t for t in tools if t.name == "pubmed_search")
    resp = json.loads(pub.func("erenumab migraine hydration"))
    assert isinstance(resp.get("total_found"), int)
    assert isinstance(resp.get("articles"), list)
    assert resp.get("query")

@pytest.mark.network
def test_clinical_trials_search_lives(tools):
    ct = next(t for t in tools if t.name == "clinical_trials_search")
    resp = json.loads(ct.func("erenumab migraine prevention"))
    assert isinstance(resp.get("trials"), list)
    assert resp.get("query")
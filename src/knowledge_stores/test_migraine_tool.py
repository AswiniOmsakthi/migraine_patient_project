# -*- coding: utf-8 -*-
# src/knowledge_stores/test_migraine_tool.py

import os
import json
import pytest

from langchain.agents import Tool
from src.knowledge_stores.migraine import load_vectorstore, migraine_country_brand_data

# ─── Fix: clear the cached LRU and reload each session ────────────────────────

@pytest.fixture(scope="module")
def tools():
    load_vectorstore.cache_clear()
    return load_vectorstore()


# ─── Test 1: All expected tools are loaded ──────────────────────────────────

def test_vectorstore_tools_loaded(tools):
    names = {t.name for t in tools}
    expected = {
        "migraine_qa_csv",
        "migraine_countries_with_on_and_off_label_usage",
        "pubmed_search",
        "clinical_trials_search",
    }
    missing = expected - names
    assert not missing, f"Missing tools: {sorted(missing)}"
    assert all(isinstance(t, Tool) for t in tools), "All elements should be LangChain Tools"


# ─── Test 2: Your CSV loader returns proper JSON with expected keys ────────

def test_country_brand_data_json():
    raw = migraine_country_brand_data("")  # interface passes an arg, but ignored internally
    arr = json.loads(raw)
    assert isinstance(arr, list), "Output must be a JSON array"
    assert arr, "List must contain at least one record"
    rec = arr[0]
    assert "Country" in rec, f"'Country' key missing: {rec}"

    # The CSV uses "Brand Name" (not "Brand"), and also should list indication columns like 'Chronic Migraine'
    brand_keys = [k for k in rec.keys() if "brand" in k.lower()]
    assert brand_keys, f"No 'brand'-containing key found in record: {list(rec.keys())}"
    assert "Chronic Migraine" in rec, f"'Chronic Migraine' column missing({list(rec.keys())})"


# ─── Test 3: QA CSV retrieval tool returns a non-empty string ──────────────

def test_qa_csv_tool_output(tools):
    qa = next(t for t in tools if t.name == "migraine_qa_csv")
    result = qa.func("migraine")  # tool invocation with "migraine" query
    assert isinstance(result, str), "QA tool should return a string"
    assert result.strip(), "QA tool output should not be blank"


# ─── Tests 4 & 5: Live PubMed and ClinicalTrials searches (never skipped) ──

def test_pubmed_search_tool_lives(tools):
    pub = next(t for t in tools if t.name == "pubmed_search")
    resp_str = pub.func("erenumab migraine hydration",)
    resp = json.loads(resp_str)
    assert isinstance(resp.get("total_found"), int), "total_found must be integer"
    assert isinstance(resp.get("articles"), list), "articles must be a list"
    assert "query" in resp and resp["query"], "Missing or empty 'query' in PubMed output"

def test_clinical_trials_search_lives(tools):
    ct = next(t for t in tools if t.name == "clinical_trials_search")
    resp_str = ct.func("erenumab migraine prevention")
    resp = json.loads(resp_str)
    assert isinstance(resp.get("trials"), list), "trials must be a list"
    assert "query" in resp and resp["query"], "Missing or empty 'query' in trials output"
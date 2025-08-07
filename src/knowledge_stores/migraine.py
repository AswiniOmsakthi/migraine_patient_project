import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from langchain.agents import Tool
from langchain.tools.retriever import create_retriever_tool
from langchain_community.vectorstores import Chroma
from Bio import Entrez  # Ensure you've installed biopython

from src.data_processing.create_db import emb_fn, build_in_memory, get_persistent_client
from src.utils.app_logger import LOGGER

_vectorstore_cache: Optional[List[Tool]] = None

class _EmbeddingAdapter:
    def __init__(self, fn: Callable[[List[str]], List[List[float]]]):
        self._fn = fn

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._fn(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._fn([text])[0]

    def __call__(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)

    def name(self) -> str:
        return getattr(self._fn, "name", "<adapter>")

@lru_cache(maxsize=1)
def migraine_country_brand_data(_: str) -> str:
    """
    Returns JSON string of list of dictionaries where each record represents
    migraine drug brand usage per country.

    Format:
      {
        "Country": "<country name>",
        "Brand Name": "<brand used>",
        "Chronic Migraine": "<Yes/No>"
      }
    """
    fp = Path(__file__).parent / "migraine_countries_with_on_and_off_label_usage.csv"
    if not fp.exists():
        LOGGER.error("CSV not found: %s", fp)
        raise FileNotFoundError(f"{fp} not found")

    df = pd.read_csv(fp, dtype=str).fillna("")

    expected_columns = ["Country", "Brand Name", "Chronic Migraine"]
    missing = [c for c in expected_columns if c not in df.columns]
    if missing:
        LOGGER.error("Missing columns in CSV: %s", missing)
        raise ValueError(f"CSV missing expected columns: {missing}")

    return df[expected_columns].to_json(orient="records")

def _build_csv_tool(mem: Any, adapter: _EmbeddingAdapter) -> Tool:
    retr = Chroma(client=mem, collection_name="migraine_QA", embedding_function=adapter)\
        .as_retriever(search_type="similarity", search_kwargs={"k": 5})
    return create_retriever_tool(
        retriever=retr,
        name="migraine_qa_csv",
        description="Retrieve migraine Q&A segments"
    )

def _build_pdf_tools(mem: Any, adapter: _EmbeddingAdapter) -> List[Tool]:
    tools = []
    for coll in mem.list_collections():
        if coll.name == "migraine_QA":
            continue
        retr = Chroma(client=mem, collection_name=coll.name, embedding_function=adapter)\
            .as_retriever(search_type="similarity", search_kwargs={"k": 5})
        tools.append(create_retriever_tool(
            retriever=retr,
            name=coll.name,
            description=f"Retrieve migraine PDF content from '{coll.name}'"
        ))
    return tools

class PubMedTool:
    def __init__(self, email: str, api_key: Optional[str] = None, rate_limit: float = 0.34):
        Entrez.email = email
        if api_key:
            Entrez.api_key = api_key
        self.rate_limit = rate_limit

    def _pause(self):
        time.sleep(self.rate_limit)

    def search_pubmed(self, query: str, max_results: int = 10, sort: str = "relevance") -> List[str]:
        try:
            handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort=sort)
            res = Entrez.read(handle)
            self._pause()
            return res.get("IdList", [])
        except Exception as e:
            LOGGER.warning("PubMed search failed: %s", e)
            return []

    def get_article_details(self, pmids: List[str]) -> List[Dict[str, Any]]:
        if not pmids:
            return []
        articles = []
        try:
            handle = Entrez.esummary(db="pubmed", id=",".join(pmids))
            summaries = Entrez.read(handle)
            self._pause()
        except Exception as e:
            LOGGER.warning("PubMed summary failed: %s", e)
            return articles
        for s in summaries:
            articles.append({
                "pmid": s.get("Id", ""),
                "title": s.get("Title", ""),
                "authors": s.get("AuthorList", [])[:3],
                "journal": s.get("Source", ""),
                "pub_date": s.get("PubDate", ""),
                "doi": s.get("DOI", ""),
                "abstract": "",
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{s.get('Id', '')}/"
            })
        return articles

    def get_abstracts(self, pmids: List[str]) -> Dict[str, str]:
        if not pmids:
            return {}
        result = {}
        try:
            handle = Entrez.efetch(db="pubmed", id=",".join(pmids), rettype="abstract", retmode="xml")
            recs = Entrez.read(handle)
            self._pause()
        except Exception as e:
            LOGGER.warning("PubMed abstracts failed: %s", e)
            return result
        for rec in recs.get("PubmedArticle", []):
            pmid = str(rec["MedlineCitation"]["PMID"])
            abst = rec["MedlineCitation"]["Article"].get("Abstract", {}).get("AbstractText", "")
            text = " ".join(abst) if isinstance(abst, list) else str(abst) or "No abstract available"
            result[pmid] = text
        return result

    def search_and_summarize(self, query: str, max_results: int = 5, include_abstracts: bool = True) -> Dict[str, Any]:
        pmids = self.search_pubmed(query, max_results)
        if not pmids:
            return {"query": query, "total_found": 0, "articles": [], "summary": f"No articles found for '{query}'."}
        arts = self.get_article_details(pmids)
        if include_abstracts:
            absmap = self.get_abstracts(pmids)
            for art in arts:
                art["abstract"] = absmap.get(art["pmid"], "")
        summary_lines = [f"Found {len(arts)} articles for '{query}':", ""]
        for idx, art in enumerate(arts, 1):
            auth = ", ".join(art["authors"]) + (" et al." if len(art["authors"]) > 3 else "")
            summary_lines.append(f"{idx}. {art['title']}\n   Authors: {auth}\n   Journal: {art['journal']} ({art['pub_date']})\n   PMID: {art['pmid']}")
            if art["abstract"]:
                abr = art["abstract"]
                summary_lines.append(f"   Abstract: {abr[:200] + '...' if len(abr) > 200 else abr}")
            summary_lines.append("")
        return {"query": query, "total_found": len(arts), "articles": arts, "summary": "\n".join(summary_lines)}

class ClinicalTrialsTool:
    def __init__(self):
        self.base_url = "https://clinicaltrials.gov/api/v2/studies"
        self.headers = {'User-Agent': 'migraine-agent', 'Content-Type': 'application/json'}

    def search_trials(self, query: str, cond: str = "", intr: str = "", max_results: int = 5) -> Dict[str, Any]:
        try:
            import requests
            resp = requests.get(self.base_url, params={
                'format': 'json',
                'query.cond': cond,
                'query.term': query,
                'pageSize': max_results,
                'countTotal': 'true'
            }, headers=self.headers, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            LOGGER.warning("ClinicalTrials search failed: %s", e)
            return {"error": str(e)}

    def search_and_summarize(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        data = self.search_trials(query=query, max_results=max_results)
        trials = []
        for s in data.get("studies", [])[:max_results]:
            ident = s.get("protocolSection", {}).get("identificationModule", {})
            status = s.get("protocolSection", {}).get("statusModule", {})
            trials.append({
                "nct_id": ident.get("nctId", ""),
                "brief_title": ident.get("briefTitle", ""),
                "overall_status": status.get("overallStatus", "")
            })
        return {"query": query, "totalCount": len(trials), "trials": trials}

def load_vectorstore(_embeddings: Optional[Any]) -> List[Tool]:
    global _vectorstore_cache
    if _vectorstore_cache is None:
        fn = _embeddings or emb_fn
        adapter = _EmbeddingAdapter(fn)
        pers = get_persistent_client()
        mem = build_in_memory(pers)

        tools: List[Tool] = [
            _build_csv_tool(mem, adapter),
            *_build_pdf_tools(mem, adapter),
            Tool(
                name="migraine_country_brand_usage",
                func=migraine_country_brand_data,
                description="Returns JSON with Country, Brand Name, Chronic Migraine columns"
            )
        ]

        pm = PubMedTool(email="you@example.com")
        tools.append(Tool(
            name="pubmed_search",
            func=lambda q: json.dumps(pm.search_and_summarize(q)),
            description="PubMed search JSON tool"
        ))

        ct = ClinicalTrialsTool()
        tools.append(Tool(
            name="clinical_trials_search",
            func=lambda q: json.dumps(ct.search_and_summarize(q)),
            description="ClinicalTrials.gov search JSON tool"
        ))

        LOGGER.info("Loaded migraine tools: %s", [t.name for t in tools])
        _vectorstore_cache = tools

    return _vectorstore_cache
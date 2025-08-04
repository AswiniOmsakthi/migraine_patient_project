# src/knowledge_stores/migraine.py

import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from langchain.agents import Tool
from langchain.tools.retriever import create_retriever_tool
from langchain_community.vectorstores import Chroma
from Bio import Entrez  # Ensure you've installed biopython

from src.data_processing.create_db import emb_fn, build_in_memory, get_persistent_client
from src.utils.app_logger import LOGGER

_vectorstore_cache: Optional[List[Tool]] = None


class _EmbeddingAdapter:
    """
    Wraps a function emb_fn(list[str]) -> List[List[float]]
    into a class with embed_documents and embed_query methods,
    so Chroma retriever can call .embed_query(...) correctly.
    """
    def __init__(self, fn: Callable[[List[str]], List[List[float]]]):
        self._fn = fn

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._fn(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._fn([text])[0]

    def name(self) -> str:
        return getattr(self._fn, "name", "<adapter>")

    def __call__(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)


def _build_csv_tool(mem: Any, adapter: _EmbeddingAdapter) -> Tool:
    retriever = Chroma(
        client=mem,
        collection_name="migraine_QA",
        embedding_function=adapter
    ).as_retriever(search_type="similarity", search_kwargs={"k": 5})

    return create_retriever_tool(
        retriever=retriever,
        name="migraine_qa_csv",
        description="Retrieve migraine Q&A segments"
    )


def _build_pdf_tools(mem: Any, adapter: _EmbeddingAdapter) -> List[Tool]:
    tools = []
    for coll in mem.list_collections():
        if coll.name == "migraine_QA":
            continue
        retr = Chroma(
            client=mem,
            collection_name=coll.name,
            embedding_function=adapter
        ).as_retriever(search_type="similarity", search_kwargs={"k": 5})
        tools.append(create_retriever_tool(
            retriever=retr,
            name=coll.name,
            description=f"Retrieve migraine PDF content from '{coll.name}'"
        ))
    return tools


class PubMedTool:
    # unchanged...
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
            handle.close()
            self._pause()
            return res.get("IdList", [])
        except Exception as e:
            LOGGER.warning("PubMed search error: %s", e)
            return []

    def get_article_details(self, pmids: List[str]) -> List[Dict[str, Any]]:
        # (same as earlier)
        if not pmids:
            return []
        try:
            handle = Entrez.esummary(db="pubmed", id=",".join(pmids))
            summaries = Entrez.read(handle)
            handle.close()
            self._pause()
        except Exception as e:
            LOGGER.warning("PubMed summary error: %s", e)
            return []
        articles = []
        for s in summaries:
            articles.append({
                "pmid":       s.get("Id", ""),
                "title":      s.get("Title", ""),
                "authors":    s.get("AuthorList", [])[:3],
                "journal":    s.get("Source", ""),
                "pub_date":   s.get("PubDate", ""),
                "doi":        s.get("DOI", ""),
                "abstract":   "",
                "url":        f"https://pubmed.ncbi.nlm.nih.gov/{s.get('Id', '')}/"
            })
        return articles

    def get_abstracts(self, pmids: List[str]) -> Dict[str, str]:
        # (same as earlier)
        if not pmids:
            return {}
        try:
            handle = Entrez.efetch(db="pubmed", id=",".join(pmids),
                                   rettype="abstract", retmode="xml")
            recs = Entrez.read(handle)
            handle.close()
            self._pause()
        except Exception as e:
            LOGGER.warning("PubMed abstracts error: %s", e)
            return {}

        result = {}
        for rec in recs.get("PubmedArticle", []):
            pmid = str(rec["MedlineCitation"]["PMID"])
            art = rec["MedlineCitation"]["Article"]
            abst = art.get("Abstract", {}).get("AbstractText", "")
            parts = []
            if isinstance(abst, list):
                for part in abst:
                    lbl = getattr(part, "attributes", {}).get("Label")
                    parts.append(f"{lbl}: {part}" if lbl else part)
            elif abst:
                parts.append(str(abst))
            result[pmid] = " ".join(parts) if parts else "No abstract available"
        return result

    def search_and_summarize(self, query: str, max_results: int = 5, include_abstracts: bool = True) -> Dict[str, Any]:
        # (same as earlier)
        pmids = self.search_pubmed(query, max_results)
        if not pmids:
            return {"query": query, "total_found": 0, "articles": [], "summary": f"No articles found for '{query}'."}

        arts = self.get_article_details(pmids)
        if include_abstracts:
            absmap = self.get_abstracts(pmids)
            for art in arts:
                art["abstract"] = absmap.get(art["pmid"], "")

        summary = [f"Found {len(arts)} articles for '{query}':", ""]
        for idx, art in enumerate(arts, 1):
            auth = ", ".join(art["authors"])
            if len(art["authors"]) > 3:
                auth += " et al."
            summary.append(
                f"{idx}. {art['title']}\n"
                f"   Authors: {auth}\n"
                f"   Journal: {art['journal']} ({art['pub_date']})\n"
                f"   PMID: {art['pmid']}"
            )
            if art.get("abstract"):
                abr = art["abstract"]
                abr = abr[:200] + ("..." if len(abr) > 200 else "")
                summary.append(f"   Abstract: {abr}")
            summary.append("")
        return {"query": query, "total_found": len(arts), "articles": arts, "summary": "\n".join(summary)}


class ClinicalTrialsTool:
    # unchanged... as before
    def __init__(self):
        self.base_url = "https://clinicaltrials.gov/api/v2/studies"
        self.headers = {'User-Agent': 'migraine-agent', 'Content-Type': 'application/json'}

    def search_trials(self, query: str, cond: str = "", intr: str = "", max_results: int = 5) -> Dict[str, Any]:
        params = {
            'format': 'json',
            'query.cond': cond,
            'query.term': query,
            'pageSize': max_results,
            'countTotal': 'true'
        }
        try:
            import requests
            resp = requests.get(self.base_url, params=params, headers=self.headers, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            LOGGER.warning("ClinicalTrials search error: %s", e)
            return {"error": str(e)}

    def search_and_summarize(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        data = self.search_trials(query=query, max_results=max_results)
        trials_list = []
        for s in data.get("studies", [])[:max_results]:
            mod = s.get("protocolSection", {})
            ident = mod.get("identificationModule", {})
            status = mod.get("statusModule", {})
            trials_list.append({
                "nct_id": ident.get("nctId", ""),
                "brief_title": ident.get("briefTitle", ""),
                "overall_status": status.get("overallStatus", "")
            })
        return {"query": query, "totalCount": len(trials_list), "trials": trials_list}


@lru_cache(maxsize=1)
def migraine_country_brand_data(_: str) -> str:
    import pandas as pd
    fp = Path(__file__).parent / "migraine_countries_with_on_and_off_label_usage.csv"
    if not fp.exists():
        raise FileNotFoundError(f"{fp} not found")
    df = pd.read_csv(fp, dtype=str).fillna("")
    return df.to_json(orient="records")


@lru_cache(maxsize=1)
def load_vectorstore(
    _embeddings: Optional[Callable[[List[str]], List[List[float]]]] = None
) -> List[Tool]:
    global _vectorstore_cache
    if _vectorstore_cache is None:
        # Use pre-existing emb_fn or override
        fn = _embeddings or emb_fn
        adapter = _EmbeddingAdapter(fn)

        pers = get_persistent_client()
        mem = build_in_memory(pers)

        tools = [
            _build_csv_tool(mem, adapter),
            *(_build_pdf_tools(mem, adapter)),
            Tool(
                name="migraine_countries_with_on_and_off_label_usage",
                func=migraine_country_brand_data,
                description="Country/Brand approval data JSON"
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
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
import contextvars
from typing import TypedDict, List, Any
from pydantic import BaseModel
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch # 🛡️ Fixed deprecation
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig 

crag_openai_key_var = contextvars.ContextVar('crag_openai_key', default="")
crag_tavily_key_var = contextvars.ContextVar('crag_tavily_key', default="")
# -----------------------------
# 1. State (Clean and Lightweight)
# -----------------------------
class CRAGState(TypedDict):
    question: str
    docs: List[Document]
    good_docs: List[Document]
    verdict: str
    reason: str
    refined_context: str # Just the raw merged text now!
    web_query: str
    web_docs: List[Document]
    answer: str

UPPER_TH = 0.7
LOWER_TH = 0.3

# -----------------------------
# 2. Nodes
# -----------------------------
async def retrieve_node(state: CRAGState, config: RunnableConfig) -> dict:
    retriever = config.get("configurable", {}).get("retriever")
    if not retriever:
        return {"docs": []}
    docs = await retriever.ainvoke(state["question"])
    return {"docs": docs}

class DocEvalScore(BaseModel):
    score: float
    reason: str

async def eval_each_doc_node(state: CRAGState, config: RunnableConfig) -> dict:
    openai_key = crag_openai_key_var.get()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_key)
    
    # 🛡️ THE FIX: Override for "Summarize" requests
    doc_eval_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a strict retrieval evaluator for RAG. Return a relevance score in [0.0, 1.0]. "
                   "1.0=sufficient, 0.0=irrelevant. "
                   "CRITICAL RULE: If the user's question asks to 'summarize', 'overview', or 'explain the document', you MUST score every chunk as 1.0. "
                   "Output JSON only."),
        ("human", "Question: {question}\n\nChunk:\n{chunk}"),
    ])
    doc_eval_chain = doc_eval_prompt | llm.with_structured_output(DocEvalScore)
    
    scores, good = [], []
    for d in state["docs"]:
        out = await doc_eval_chain.ainvoke({"question": state["question"], "chunk": d.page_content})
        scores.append(out.score)
        if out.score > LOWER_TH: good.append(d)

    if any(s > UPPER_TH for s in scores):
        return {"good_docs": good, "verdict": "CORRECT", "reason": "Doc scored > UPPER_TH"}
    if len(scores) > 0 and all(s < LOWER_TH for s in scores):
        return {"good_docs": [], "verdict": "INCORRECT", "reason": "All docs < LOWER_TH"}
    
    return {"good_docs": good, "verdict": "AMBIGUOUS", "reason": "Mixed scores"}


# 🛡️ THE NEW SPEED NODE: Just merges the text instantly
async def merge_context(state: CRAGState) -> dict:
    if state.get("verdict") == "CORRECT":
        docs_to_use = state.get("good_docs", [])
    elif state.get("verdict") == "INCORRECT":
        docs_to_use = state.get("web_docs", [])
    else: # AMBIGUOUS
        docs_to_use = state.get("good_docs", []) + state.get("web_docs", [])

    refined_context = "\n\n".join(d.page_content for d in docs_to_use).strip()
    return {"refined_context": refined_context}


class WebQuery(BaseModel):
    query: str

async def rewrite_query_node(state: CRAGState, config: RunnableConfig) -> dict:
    openai_key = crag_openai_key_var.get()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_key)
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", "Rewrite the user question into a web search query (6-14 words). Return JSON only."),
        ("human", "Question: {question}"),
    ])
    out = await (rewrite_prompt | llm.with_structured_output(WebQuery)).ainvoke({"question": state["question"]})
    return {"web_query": out.query}

async def web_search_node(state: CRAGState, config: RunnableConfig) -> dict:
    tavily_key = crag_tavily_key_var.get()
    if not tavily_key:
        return {"web_docs": [Document(page_content="Error: No Tavily API Key provided.")]}
        
    tavily = TavilySearch(max_results=3, tavily_api_key=tavily_key)
    q = state.get("web_query") or state["question"]
    raw_results = await tavily.ainvoke({"query": q}) 
    
    # 🛡️ THE FIX: Safely unpack the new dictionary format
    if isinstance(raw_results, dict) and "results" in raw_results:
        results_list = raw_results["results"]
    elif isinstance(raw_results, list):
        results_list = raw_results
    else:
        results_list = [{"content": str(raw_results)}] # Fallback for weird errors
        
    web_docs = []
    for r in results_list:
        if isinstance(r, dict):
            text = f"TITLE: {r.get('title', '')}\nCONTENT:\n{r.get('content', '')}"
            url = r.get("url", "")
            web_docs.append(Document(page_content=text, metadata={"url": url}))
            
    return {"web_docs": web_docs}

async def generate(state: CRAGState, config: RunnableConfig) -> dict:
    openai_key = crag_openai_key_var.get()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_key)
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Answer ONLY using the provided context to answer in well structured format. If insufficient, say 'I don't know.'"),
        ("human", "Question: {question}\n\nContext:\n{context}"),
    ])
    out = await (answer_prompt | llm).ainvoke({"question": state["question"], "context": state["refined_context"]})
    return {"answer": out.content}

def route_after_eval(state: CRAGState) -> str:
    return "merge_context" if state["verdict"] == "CORRECT" else "rewrite_query"

# -----------------------------
# 3. Compile Graph
# -----------------------------
crag_builder = StateGraph(CRAGState)
crag_builder.add_node("retrieve", retrieve_node)
crag_builder.add_node("eval_each_doc", eval_each_doc_node)
crag_builder.add_node("rewrite_query", rewrite_query_node)
crag_builder.add_node("web_search", web_search_node)
crag_builder.add_node("merge_context", merge_context) # 🛡️ Replaced "refine"
crag_builder.add_node("generate", generate)

crag_builder.add_edge(START, "retrieve")
crag_builder.add_edge("retrieve", "eval_each_doc")
crag_builder.add_conditional_edges("eval_each_doc", route_after_eval)
crag_builder.add_edge("rewrite_query", "web_search")
crag_builder.add_edge("web_search", "merge_context") # 🛡️ Routes to merge
crag_builder.add_edge("merge_context", "generate") # 🛡️ Routes to generate
crag_builder.add_edge("generate", END)

crag_app = crag_builder.compile()
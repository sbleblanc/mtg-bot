from typing import List, Dict

import dspy
import streamlit as st
from dsp.utils.utils import deduplicate
from fastembed import TextEmbedding, LateInteractionTextEmbedding
from fastembed.sparse.bm25 import Bm25
from qdrant_client import QdrantClient
from mtg_bot.retrievers import QdrantPrefetchRM


class SearchQuery(dspy.Signature):
    """Generate query about Magic the Gathering rules and cards"""
    context: List[str] = dspy.InputField(desc="rules and card descriptions")
    question: str = dspy.InputField()
    query: str = dspy.OutputField()


class QuestionAnswering(dspy.Signature):
    """Answer questions about Magic the Gathering rules and cards"""
    context: List[str] = dspy.InputField(desc="rules and card descriptions")
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="the answer to the question listing all the potential cases.")
    game_mechanics: Dict[str, str] = dspy.OutputField(desc="the game mechanics involved and their implication in the answer")


@st.cache_resource
def get_prefetch_qdrant_retriever(
        qdrant_client_db_path: str,
        collection_name: str,
        text_emb_model_name: str,
        late_emb_model_name: str,
        bm25_model_name: str,
        cuda: bool = False
) -> QdrantPrefetchRM:
    return QdrantPrefetchRM(
        QdrantClient(path=qdrant_client_db_path),
        collection_name,
        TextEmbedding(text_emb_model_name, cuda=cuda),
        LateInteractionTextEmbedding(late_emb_model_name, cuda=cuda),
        Bm25(bm25_model_name)
    )


retriever = get_prefetch_qdrant_retriever(
    "data/mtg_db",
    "mtg",
    "BAAI/bge-small-en-v1.5",
    "jinaai/jina-colbert-v2",
    "Qdrant/bm25"
)

with st.sidebar:
    temp_slider = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.05
    )

    ret_k_slider = st.slider(
        "Retreiver k",
        min_value=1,
        max_value=50,
        value=10,
        step=1
    )

    num_hops_slider = st.slider(
        "Number of Hops",
        min_value=0,
        max_value=5,
        value=0,
        step=1
    )

    per_hop_ret_k_slider = st.slider(
        "Per-hop retriever k",
        min_value=1,
        max_value=50,
        value=3,
        step=1
    )

lm = dspy.LM(
    'custom_openai/phi-3.5',
    api_base="http://127.0.0.1:8080",
    api_key='123',
    temperature=temp_slider
)
dspy.configure(lm=lm)
gen_queries = [dspy.ChainOfThought(SearchQuery) for _ in range(num_hops_slider)]
gen_answer = dspy.ChainOfThought(QuestionAnswering)

st.title("MTG Rag Prototype")

if question_prompt := st.chat_input("Ask a question about MTG"):

    with st.chat_message("user"):
        st.markdown(question_prompt)

    if len(gen_queries) > 0:
        with st.chat_message("assistant"):
            st.markdown("# Multi-hop Reasoning")
            context = []
            for i, hop_gen_query in enumerate(gen_queries):
                st.markdown(f"## Hop {i+1}")
                search_query_res = hop_gen_query(context=context, question=question_prompt, k=ret_k_slider)
                passages = retriever(search_query_res.query, k=per_hop_ret_k_slider)
                context = deduplicate(passages + context)
                st.markdown("### Query")
                st.markdown(search_query_res.query)
                st.markdown("### Rationale")
                st.markdown(search_query_res.reasoning)
                with st.expander("Context"):
                    for j, c in enumerate(context):
                        st.markdown(f"{j + 1}. {c}")

    if num_hops_slider == 0:
        context = retriever(question_prompt, k=ret_k_slider)

    answer_res = gen_answer(context=context, question=question_prompt)

    with st.chat_message("assistant"):
        st.markdown("# Answer")
        st.markdown(answer_res.answer)

        st.markdown("# Rationale")
        st.markdown(answer_res.reasoning)

        st.markdown("# Game Mechanics")
        st.markdown(answer_res.game_mechanics)

        with st.expander("Context"):
            for i, c in enumerate(context):
                st.markdown(f"{i + 1}. {c}")


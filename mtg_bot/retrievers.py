from collections import defaultdict
from typing import Optional, Union, List

import dspy
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, LateInteractionTextEmbedding
from fastembed.sparse.bm25 import Bm25


class QdrantPrefetchRM(dspy.Retrieve):

    def __init__(
            self,
            q_client: QdrantClient,
            collection_name: str,
            text_emb_model: TextEmbedding,
            late_emb_model: LateInteractionTextEmbedding,
            bm25_model: Bm25,
            text_emb_name: str = "text_emb",
            sparse_emb_name: str = "bm25",
            late_emb_name: str = "late_emb",
            text_emb_k: int = 50,
            sparse_emb_k: int = 50,
            late_emb_k: int = 10,
            document_field: str = "text"
    ):
        self._client = q_client
        self._collection_name = collection_name

        self._text_emb_model = text_emb_model
        self._late_emb_model = late_emb_model
        self._bm25_model = bm25_model

        self._text_emb_name = text_emb_name
        self._sparse_emb_name = sparse_emb_name
        self._late_emb_name = late_emb_name
        self._text_emb_k = text_emb_k
        self._sparse_emb_k = sparse_emb_k
        self._late_emb_k = late_emb_k
        self._document_field = document_field

        super().__init__(k=late_emb_k)

    def __build_query(self, query_str: str, k: Optional[int] = None) -> models.QueryRequest:
        text_emb_prefetch = models.Prefetch(
            query=next(self._text_emb_model.query_embed(query_str)),
            using=self._text_emb_name,
            limit=self._text_emb_k
        )

        bm_25_embedded = next(self._bm25_model.query_embed(query_str))

        sparse_emb_prefetch = models.Prefetch(
            query=models.SparseVector(**bm_25_embedded.as_object()),
            using=self._sparse_emb_name,
            limit=self._sparse_emb_k
        )

        return models.QueryRequest(
            prefetch=[text_emb_prefetch, sparse_emb_prefetch],
            query=next(self._late_emb_model.query_embed(query_str)),
            using=self._late_emb_name,
            with_payload=True,
            limit=k if k else self._late_emb_k
        )

    def forward(
            self,
            query_or_queries: Union[str, List[str]] = None,
            query: Optional[str] = None,
            k: Optional[int] = None,
            by_prob: bool = True,
            with_metadata: bool = False,
            **kwargs,
    ) -> Union[List[str], dspy.Prediction, List[dspy.Prediction]]:
        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries
        queries = [q for q in queries if q]  # Filter empty queries

        q_reqs = list(map(self.__build_query, queries))

        passages_scores = defaultdict(float)
        for batch_res in self._client.query_batch_points(self._collection_name, q_reqs):
            for res_point in batch_res.points:
                document = res_point.payload[self._document_field]
                passages_scores[document] += res_point.score

        sorted_passages = sorted(passages_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [passage for passage, _ in sorted_passages]


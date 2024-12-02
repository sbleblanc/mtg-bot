import click
import json
import tqdm
import uuid
from typing import Tuple
from pathlib import Path
from mtg_bot.preprocessing.mtg_rules import rules_to_documents
from dvc.api import params_show
from fastembed import TextEmbedding, LateInteractionTextEmbedding
from fastembed.sparse.bm25 import Bm25
from qdrant_client import QdrantClient, models


@click.group()
def cli():
    pass


@cli.command()
@click.argument("src", nargs=1, type=click.Path(dir_okay=False, path_type=Path))
@click.argument("dst", nargs=1, type=click.Path(dir_okay=False, path_type=Path))
def mtg_rules(src: Path, dst: Path):
    with src.open() as f:
        documents = [
            {
                "id": str(uuid.uuid4()),
                "text": d
            }
            for d in rules_to_documents(f)
        ]

    with dst.open('w') as f:
        json.dump(documents, f, indent=2)


@cli.command()
@click.argument("src", nargs=1, type=click.Path(dir_okay=False, path_type=Path))
@click.argument("dst", nargs=1, type=click.Path(dir_okay=False, path_type=Path))
def build_embeddings(
        src: Path,
        dst: Path
):
    params = params_show()
    text_emb_model = TextEmbedding(params['embeddings']['text_emb_model'], cuda=True)
    late_emb_model = LateInteractionTextEmbedding(params['embeddings']['late_emb_model'], cuda=True)
    bm25_model = Bm25(params['embeddings']['bm25_model'])

    with src.open() as f:
        documents = json.load(f)

    text_emb = text_emb_model.embed((d['text'] for d in documents), batch_size=2)
    late_emb = late_emb_model.embed((d['text'] for d in documents), batch_size=2)
    bm25_emb = bm25_model.embed((d['text'] for d in documents), batch_size=2)

    for d, te, le, se in tqdm.tqdm(zip(documents, text_emb, late_emb, bm25_emb), total=len(documents), desc="Embedding docs..."):
        d['text_emb'] = te.tolist()
        d['late_emb'] = le.tolist()
        d['sparse_emb'] = {
            "indices": se.indices.tolist(),
            "values": se.values.tolist()
        }

    with dst.open('w') as f:
        json.dump(documents, f, indent=2)


@cli.command()
@click.argument("db_dst", nargs=1, type=click.Path(dir_okay=False))
@click.argument("src", nargs=-1, type=click.Path(dir_okay=False, path_type=Path))
def build_qdrant_db(
        db_dst: str,
        src: Tuple[Path]
):
    documents = []

    for p in src:
        with p.open() as f:
            documents.extend(json.load(f))

    client = QdrantClient(path=db_dst)
    client.create_collection(
        "mtg",
        vectors_config={
            "text_emb": models.VectorParams(
                size=len(documents[0]['text_emb']),
                distance=models.Distance.COSINE
            ),
            "late_emb": models.VectorParams(
                size=len(documents[0]['late_emb'][0]),
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                )
            ),
        },
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(
                modifier=models.Modifier.IDF
            )
        }
    )

    for d in tqdm.tqdm(documents, desc="Uploading data", unit="document"):
        client.upload_points(
            "mtg",
            points=[
                models.PointStruct(
                    id=d['id'],
                    vector={
                        "text_emb": d['text_emb'],
                        "late_emb": d['late_emb'],
                        "bm25": d['sparse_emb']
                    },
                    payload={
                        "text": d['text']
                    }
                )
            ],
        )



if __name__ == "__main__":
    cli()

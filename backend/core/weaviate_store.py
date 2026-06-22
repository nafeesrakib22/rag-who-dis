

import uuid
import weaviate
import weaviate.classes as wvc
from . import config


class WeaviateStore:


    def __init__(self):
        """
        Connect to a standalone Weaviate instance running in Docker.
        """
        host = config.WEAVIATE_HOST
        port = config.WEAVIATE_PORT
        grpc_port = config.WEAVIATE_GRPC_PORT

        print(f"[weaviate_store] Connecting to Weaviate at {host}:{port}...")

        try:
            self.client = weaviate.connect_to_local(
                host=host,
                port=port,
                grpc_port=grpc_port,
                additional_config=wvc.init.AdditionalConfig(
                    timeout=wvc.init.Timeout(init=60, query=30, insert=120)
                ),
            )
        except Exception as e:
            raise RuntimeError(
                f"Could not connect to Weaviate at {host}:{port}.\n"
                "Make sure your Docker container is running: docker compose up -d"
            ) from e

        print(f"[weaviate_store] Connected. Ensuring collection '{config.WEAVIATE_COLLECTION_NAME}' exists...")
        self._ensure_collection()

        count = self.count()
        print(f"[weaviate_store] Collection ready. Current object count: {count}")

    def _ensure_collection(self):

        if not self.client.collections.exists(config.WEAVIATE_COLLECTION_NAME):
            self.client.collections.create(
                name=config.WEAVIATE_COLLECTION_NAME,
                vectorizer_config=wvc.config.Configure.Vectorizer.none(),
                properties=[
                    wvc.config.Property(
                        name="text",
                        data_type=wvc.config.DataType.TEXT,
                        tokenization=wvc.config.Tokenization.WORD,
                    ),
                    wvc.config.Property(
                        name="source",
                        data_type=wvc.config.DataType.TEXT,
                        skip_vectorization=True,
                        tokenization=wvc.config.Tokenization.FIELD,
                    ),
                    wvc.config.Property(
                        name="page",
                        data_type=wvc.config.DataType.INT,
                    ),
                    wvc.config.Property(
                        name="chunk_index",
                        data_type=wvc.config.DataType.INT,
                    ),
                ],
            )
            print(f"[weaviate_store] Created collection '{config.WEAVIATE_COLLECTION_NAME}'.")

        self.collection = self.client.collections.get(config.WEAVIATE_COLLECTION_NAME)

    @staticmethod
    def _make_uuid(source: str, page: int, chunk_index: int) -> str:
        """
        Generate a deterministic UUID for a chunk based on its identity.

        """
        identity = f"{source}__p{page}__c{chunk_index}"
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, identity))

    def add_chunks(self, chunks: list[dict], embeddings: list[list[float]]) -> None:

        print(f"[weaviate_store] Upserting {len(chunks)} chunks...")

        with self.collection.batch.dynamic() as batch:
            for chunk, embedding in zip(chunks, embeddings):
                obj_uuid = self._make_uuid(
                    chunk["source"], chunk["page"], chunk["chunk_index"]
                )
                batch.add_object(
                    properties={
                        "text":          chunk["text"],
                        "source":        chunk["source"],
                        "page":          chunk["page"],
                        "chunk_index":   chunk["chunk_index"],
                    },
                    vector=embedding,
                    uuid=obj_uuid,
                )

        # Check for any batch errors
        if self.collection.batch.failed_objects:
            n = len(self.collection.batch.failed_objects)
            print(f"[weaviate_store] Warning: {n} objects failed to insert.")

        total = self.count()
        print(f"[weaviate_store] Done. Total objects in DB: {total}")

    def query(
        self,
        query_vector: list[float],
        n_results: int,
        query_text: str = None,
        filters=None,
    ) -> list[dict]:
        
        if query_text:
            response = self.collection.query.hybrid(
                query=query_text,
                vector=query_vector,
                alpha=config.HYBRID_ALPHA,
                filters=filters,
                limit=n_results,
                return_metadata=wvc.query.MetadataQuery(score=True),
            )
            score_key = "score"
        else:
            response = self.collection.query.near_vector(
                near_vector=query_vector,
                filters=filters,
                limit=n_results,
                return_metadata=wvc.query.MetadataQuery(distance=True),
            )
            score_key = None

        results = []
        for obj in response.objects:
            props = obj.properties
            if score_key and obj.metadata.score is not None:
                hybrid_score = round(float(obj.metadata.score), 4)
            else:
                hybrid_score = None

            distance = obj.metadata.distance if (not score_key and obj.metadata.distance) else None

            results.append({
                "text":        props["text"],
                "source":      props["source"],
                "page":        int(props["page"]),
                "chunk_index": int(props["chunk_index"]),
                # Use distance if available, else convert hybrid score (higher=better → lower=closer)
                "distance":    round(distance, 4) if distance else (
                    round(1 - hybrid_score, 4) if hybrid_score is not None else 0.0
                ),
                "hybrid_score": hybrid_score,
            })

        return results

    def count(self) -> int:
        """Return the number of objects stored in the collection."""
        agg = self.collection.aggregate.over_all(total_count=True)
        return agg.total_count or 0

    def source_exists(self, source_name: str) -> bool:
        """Return True if at least one chunk for this source filename exists."""
        response = self.collection.query.fetch_objects(
            filters=wvc.query.Filter.by_property("source").equal(source_name),
            limit=1,
            return_properties=["source"],
        )
        return len(response.objects) > 0

    def get_sources(self) -> list[str]:
        """Return a sorted list of distinct source filenames in the store."""
        response = self.collection.query.fetch_objects(
            return_properties=["source"],
            limit=10_000,
        )
        return sorted({obj.properties["source"] for obj in response.objects})

    def clear(self) -> None:
        """Delete the entire collection (irreversible)."""
        self.client.collections.delete(config.WEAVIATE_COLLECTION_NAME)
        print(f"[weaviate_store] Collection '{config.WEAVIATE_COLLECTION_NAME}' deleted.")
        # Recreate empty collection
        self._ensure_collection()

    def close(self) -> None:
        """
        Cleanly shut down the embedded Weaviate instance.
        Call this when done — especially important for Embedded mode.
        """
        self.client.close()
        print("[weaviate_store] Weaviate connection closed.")

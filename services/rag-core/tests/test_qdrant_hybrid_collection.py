from __future__ import annotations

from types import SimpleNamespace

from app.clients.qdrant import DENSE_VECTOR_NAME, ensure_hybrid_collection, mark_superseded


class _FakeQdrantClient:
    def __init__(self, existing_vectors=None):
        self._exists = existing_vectors is not None
        self._vectors = existing_vectors
        self.deleted = []
        self.created = []
        self.set_payload_calls = []
        self.indexed_fields = []

    def collection_exists(self, name):
        return self._exists

    def get_collection(self, name):
        return SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors=self._vectors)))

    def delete_collection(self, name):
        self.deleted.append(name)
        self._exists = False

    def create_collection(self, collection_name, vectors_config, sparse_vectors_config=None):
        self.created.append((collection_name, vectors_config, sparse_vectors_config))
        self._exists = True
        self._vectors = vectors_config

    def set_payload(self, collection_name, payload, points, **kwargs):
        self.set_payload_calls.append((collection_name, payload, points))

    def create_payload_index(self, collection_name, field_name, field_schema):
        self.indexed_fields.append((collection_name, field_name, field_schema))


def test_creates_a_new_hybrid_collection_when_none_exists():
    client = _FakeQdrantClient(existing_vectors=None)

    recreated = ensure_hybrid_collection(client, "knowledge_base", 1024)

    assert recreated is False
    assert len(client.created) == 1
    name, vectors_config, sparse_config = client.created[0]
    assert name == "knowledge_base"
    assert DENSE_VECTOR_NAME in vectors_config
    assert sparse_config is not None
    assert client.deleted == []


def test_leaves_an_existing_hybrid_collection_untouched():
    client = _FakeQdrantClient(existing_vectors={DENSE_VECTOR_NAME: object()})

    recreated = ensure_hybrid_collection(client, "knowledge_base", 1024)

    assert recreated is False
    assert client.created == []
    assert client.deleted == []


def test_recreates_a_collection_built_under_the_old_unnamed_vector_schema():
    # Stage 0-4 schema: a single unnamed vector (not a dict keyed by name).
    client = _FakeQdrantClient(existing_vectors=object())

    recreated = ensure_hybrid_collection(client, "knowledge_base", 1024)

    assert recreated is True
    assert client.deleted == ["knowledge_base"]
    assert len(client.created) == 1


def test_creates_a_payload_index_for_every_field_ever_filtered_on():
    # Regression: Qdrant Cloud rejects a filter on an unindexed field
    # ("Index required but not found") even though self-hosted Qdrant
    # allows it via a full scan - live-caught on the very first ingest
    # against a real Qdrant Cloud cluster.
    client = _FakeQdrantClient(existing_vectors=None)

    ensure_hybrid_collection(client, "knowledge_base", 1024)

    indexed = {field for _, field, _ in client.indexed_fields}
    assert indexed == {"doc_id", "source_filename", "doc_version", "ingested_at", "superseded"}
    assert all(collection == "knowledge_base" for collection, _, _ in client.indexed_fields)


def test_backfills_payload_indexes_on_an_already_correct_existing_collection():
    # A collection that predates this fix (already the right vector schema)
    # must still get indexed on its next ensure_hybrid_collection call, not
    # just on first creation - no one-off migration script required.
    client = _FakeQdrantClient(existing_vectors={DENSE_VECTOR_NAME: object()})

    ensure_hybrid_collection(client, "knowledge_base", 1024)

    indexed = {field for _, field, _ in client.indexed_fields}
    assert indexed == {"doc_id", "source_filename", "doc_version", "ingested_at", "superseded"}


def test_mark_superseded_sets_the_flag_via_a_doc_id_filter():
    client = _FakeQdrantClient(existing_vectors={DENSE_VECTOR_NAME: object()})

    mark_superseded(client, "knowledge_base", "old-doc-id")

    collection, payload, points = client.set_payload_calls[0]
    assert collection == "knowledge_base"
    assert payload == {"superseded": True}
    assert points.must[0].key == "doc_id"
    assert points.must[0].match.value == "old-doc-id"

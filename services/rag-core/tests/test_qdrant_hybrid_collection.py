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


def test_mark_superseded_sets_the_flag_via_a_doc_id_filter():
    client = _FakeQdrantClient(existing_vectors={DENSE_VECTOR_NAME: object()})

    mark_superseded(client, "knowledge_base", "old-doc-id")

    collection, payload, points = client.set_payload_calls[0]
    assert collection == "knowledge_base"
    assert payload == {"superseded": True}
    assert points.must[0].key == "doc_id"
    assert points.must[0].match.value == "old-doc-id"

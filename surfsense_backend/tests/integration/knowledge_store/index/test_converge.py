"""Postgres converges to the store: real git engine, real Redis lock, real DB.

The indexer's whole job is a projection, so these tests assert on the projection
(rows, ids, chunk ids, the drift stamp) rather than on how it got there.

Paths carry ``.xml`` because that is what every real writer produces —
``safe_filename`` appends it — and the extension is what ``parse_documents_path``
strips to recover a title.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import func, select

from app.config import config as app_config
from app.db import (
    Chunk,
    Document,
    DocumentStatus,
    DocumentType,
    DocumentVersion,
    Folder,
)
from app.indexing_pipeline.connector_document import ConnectorDocument
from app.indexing_pipeline.indexing_pipeline_service import IndexingPipelineService
from app.knowledge_store import KnowledgeStore
from app.knowledge_store.identities import AGENT_IDENTITY, user_identity
from app.knowledge_store.index.converge import index_changes, index_tree
from app.knowledge_store.service import record_deleted_documents
from app.utils.document_converters import generate_unique_identifier_hash

pytestmark = pytest.mark.integration

# Content for the move tests. Git recognises a moved file by its content, so what
# matters is that the same bytes land at the new path.
MOVABLE = "# Content\n\na body git can match at its new path\n"


@pytest.fixture
def knowledge_root(tmp_path, monkeypatch):
    monkeypatch.setattr(app_config, "KNOWLEDGE_STORE_ENABLED", True)
    monkeypatch.setattr(app_config, "KNOWLEDGE_STORE_ROOT", str(tmp_path))
    return tmp_path


@pytest.fixture
def store(knowledge_root, db_workspace):
    return KnowledgeStore.for_workspace(db_workspace.id)


async def commit(store, writes=None, removes=(), author=None):
    """Record one revision; ``writes`` maps store path to markdown."""
    async with store.transaction(
        message="test", author=author or user_identity("1")
    ) as tx:
        for path, markdown in (writes or {}).items():
            tx.write(path, markdown.encode())
        for path in removes:
            tx.remove(path)
    return tx.revision


async def titles(session, workspace_id) -> dict[str, Document]:
    """Every document in the workspace, keyed by title."""
    result = await session.execute(
        select(Document).where(Document.workspace_id == workspace_id)
    )
    return {document.title: document for document in result.scalars()}


async def by_path(session, workspace_id) -> dict[str, Document]:
    """Every document keyed by the path it lives at.

    A move keeps the Postgres-owned title, so two rows can share a title while
    living at different paths; the path is what stays unique across a move.
    """
    result = await session.execute(
        select(Document).where(Document.workspace_id == workspace_id)
    )
    return {document.path: document for document in result.scalars()}


async def chunk_ids(session, document_id) -> list[int]:
    result = await session.execute(
        select(Chunk.id).where(Chunk.document_id == document_id).order_by(Chunk.id)
    )
    return list(result.scalars())


async def versions(session, document_id) -> list[int]:
    result = await session.execute(
        select(DocumentVersion.version_number)
        .where(DocumentVersion.document_id == document_id)
        .order_by(DocumentVersion.version_number)
    )
    return list(result.scalars())


async def chunk_shapes(session, document_id) -> list[tuple]:
    """Text and line span of every chunk, in document order."""
    result = await session.execute(
        select(Chunk.content, Chunk.start_line, Chunk.end_line)
        .where(Chunk.document_id == document_id)
        .order_by(Chunk.position)
    )
    return list(result.all())


# ── Identity ────────────────────────────────────────────────────────────────


async def test_identical_content_at_two_paths_yields_two_documents(
    store, db_session, db_workspace, patched_embed_texts
):
    """The case prepare_for_indexing collapses; `cp a b` is legal git."""
    await commit(store, {"documents/a.xml": "# Same", "documents/b.xml": "# Same"})

    await index_changes(db_session, db_workspace.id)

    rows = await titles(db_session, db_workspace.id)
    assert set(rows) == {"a", "b"}
    assert rows["a"].id != rows["b"].id


async def test_uploaded_file_is_adopted_not_duplicated(
    store, db_session, db_workspace, db_user, patched_embed_texts
):
    """An upload already has a row under its own identity; indexing must reuse it."""
    upload = Document(
        title="report.pdf",
        document_type=DocumentType.FILE,
        document_metadata={"FILE_NAME": "report.pdf"},
        content="# Report",
        content_hash=f"hash-{uuid.uuid4().hex}",
        unique_identifier_hash=generate_unique_identifier_hash(
            DocumentType.FILE, "report.pdf", db_workspace.id
        ),
        source_markdown="# Report",
        workspace_id=db_workspace.id,
        created_by_id=db_user.id,
        status=DocumentStatus.ready(),
    )
    db_session.add(upload)
    await db_session.flush()
    upload_id, upload_hash = upload.id, upload.unique_identifier_hash

    await commit(store, {"documents/report.pdf.xml": "# Report"})
    await index_changes(db_session, db_workspace.id)

    total = await db_session.scalar(
        select(func.count(Document.id)).where(Document.workspace_id == db_workspace.id)
    )
    assert total == 1
    await db_session.refresh(upload)
    assert upload.id == upload_id
    # Identity and type stay the upload's; only the location path is added.
    assert upload.document_type == DocumentType.FILE
    assert upload.unique_identifier_hash == upload_hash
    assert upload.path == "/documents/report.pdf.xml"
    assert upload.document_metadata["FILE_NAME"] == "report.pdf"


async def test_reindexing_leaves_a_rows_metadata_untouched(
    store, db_session, db_workspace, db_user, patched_embed_texts
):
    """A content update must not rewrite metadata a connector or upload owns.

    The existing-row branch of the upsert once reassigned document_metadata to
    carry the path marker; the path column now records the location, so a reindex
    has no reason to touch metadata — and must not resurrect the marker.
    """
    metadata = {"FILE_NAME": "report.pdf", "ETAG": "v1"}
    upload = Document(
        title="report.pdf",
        document_type=DocumentType.FILE,
        document_metadata=dict(metadata),
        content="# Report",
        content_hash=f"hash-{uuid.uuid4().hex}",
        unique_identifier_hash=generate_unique_identifier_hash(
            DocumentType.FILE, "report.pdf", db_workspace.id
        ),
        source_markdown="# Report",
        workspace_id=db_workspace.id,
        created_by_id=db_user.id,
        status=DocumentStatus.ready(),
    )
    db_session.add(upload)
    await db_session.flush()

    await commit(store, {"documents/report.pdf.xml": "# Report"})
    await index_changes(db_session, db_workspace.id)

    # A real content change forces the existing-row branch to run a second time.
    await commit(store, {"documents/report.pdf.xml": "# Report\n\nRevised."})
    await index_changes(db_session, db_workspace.id)

    await db_session.refresh(upload)
    assert "Revised." in upload.source_markdown
    # Exact equality proves the branch neither wiped the dict nor stamped a marker.
    assert upload.document_metadata == metadata


async def test_existing_path_updates_in_place(
    store, db_session, db_workspace, patched_embed_texts
):
    """A second index of the same path is an update, not a unique-hash collision."""
    await commit(store, {"documents/note.xml": "# First"})
    await index_changes(db_session, db_workspace.id)
    first_id = (await titles(db_session, db_workspace.id))["note"].id

    await commit(store, {"documents/note.xml": "# First\n\nSecond paragraph."})
    await index_changes(db_session, db_workspace.id)

    rows = await titles(db_session, db_workspace.id)
    assert len(rows) == 1
    assert rows["note"].id == first_id
    assert "Second paragraph." in rows["note"].source_markdown


async def test_a_document_in_a_folder_lands_under_that_folder(
    store, db_session, db_workspace, patched_embed_texts
):
    await commit(store, {"documents/Research/paper.xml": "# Paper"})

    await index_changes(db_session, db_workspace.id)

    row = (await titles(db_session, db_workspace.id))["paper"]
    assert row.folder_id is not None
    assert row.path == "/documents/Research/paper.xml"


# ── Chunk reuse and removal ─────────────────────────────────────────────────


async def test_editing_one_file_leaves_another_files_chunks_untouched(
    store, db_session, db_workspace, patched_embed_texts
):
    await commit(
        store, {"documents/a.xml": "# A\n\nAlpha.", "documents/b.xml": "# B\n\nBeta."}
    )
    await index_changes(db_session, db_workspace.id)
    untouched = (await titles(db_session, db_workspace.id))["b"]
    before = await chunk_ids(db_session, untouched.id)
    assert before

    await commit(store, {"documents/a.xml": "# A\n\nAlpha, revised."})
    await index_changes(db_session, db_workspace.id)

    assert await chunk_ids(db_session, untouched.id) == before


async def test_removed_path_deletes_the_document_and_its_chunks(
    store, db_session, db_workspace, patched_embed_texts
):
    await commit(store, {"documents/a.xml": "# A", "documents/b.xml": "# B"})
    await index_changes(db_session, db_workspace.id)
    doomed_id = (await titles(db_session, db_workspace.id))["a"].id
    assert await chunk_ids(db_session, doomed_id)

    await commit(store, removes=["documents/a.xml"])
    await index_changes(db_session, db_workspace.id)

    assert set(await titles(db_session, db_workspace.id)) == {"b"}
    assert await chunk_ids(db_session, doomed_id) == []


async def test_a_move_keeps_the_document_and_its_history(
    store, db_session, db_workspace, patched_embed_texts
):
    """The row has to outlive a move, not just be replaced by an equivalent one.
    Its version history cascades from the id, so a new row means an agent moving
    a file silently destroys every saved version of it — and dangles the
    ``document_id`` in citations already written into past answers."""
    await commit(store, {"documents/old.xml": MOVABLE})
    await index_changes(db_session, db_workspace.id)
    document_id = (await titles(db_session, db_workspace.id))["old"].id
    db_session.add(
        DocumentVersion(
            document_id=document_id,
            version_number=1,
            source_markdown=MOVABLE,
            content_hash=f"hash-{uuid.uuid4().hex}",
            title="old",
        )
    )
    await db_session.commit()

    await commit(store, {"documents/new.xml": MOVABLE}, removes=["documents/old.xml"])
    await index_changes(db_session, db_workspace.id)

    # Title is Postgres-owned, so a move keeps it; only the path follows the file.
    rows = await by_path(db_session, db_workspace.id)
    assert set(rows) == {"/documents/new.xml"}
    assert rows["/documents/new.xml"].id == document_id
    assert await versions(db_session, document_id) == [1]


async def test_a_new_file_at_a_moved_from_path_gets_its_own_row(
    store, db_session, db_workspace, patched_embed_texts
):
    """The moved row's fallback identity has to travel with it. Left behind on the
    old path, it makes the next document written there resolve to the moved row —
    one row claiming two paths, and the new file never getting a row of its own."""
    await commit(store, {"documents/old.xml": MOVABLE})
    await index_changes(db_session, db_workspace.id)
    moved_id = (await titles(db_session, db_workspace.id))["old"].id

    await commit(store, {"documents/new.xml": MOVABLE}, removes=["documents/old.xml"])
    await index_changes(db_session, db_workspace.id)
    await commit(store, {"documents/old.xml": "# Reused path\n\nnew note here\n"})
    await index_changes(db_session, db_workspace.id)

    rows = await by_path(db_session, db_workspace.id)
    assert set(rows) == {"/documents/new.xml", "/documents/old.xml"}
    assert rows["/documents/new.xml"].id == moved_id
    assert rows["/documents/old.xml"].id != moved_id


async def test_a_move_that_rewrites_the_file_keeps_the_relocated_document(
    store, db_session, db_workspace, patched_embed_texts
):
    """With nothing in common git sees no move, so the halves arrive separately —
    and an editor retitle has already moved the path column while leaving
    unique_identifier_hash behind. The removal then resolves by that hash to the
    row the upsert just updated, dropping a document whose file is still in the
    tree, invisible until the next full rebuild."""
    await commit(store, {"documents/old.xml": MOVABLE})
    await index_changes(db_session, db_workspace.id)
    row = (await titles(db_session, db_workspace.id))["old"]
    document_id = row.id
    row.path = "/documents/new.xml"
    await db_session.commit()

    await commit(
        store,
        {"documents/new.xml": "# Other\n\nnothing whatever in common\n"},
        removes=["documents/old.xml"],
    )
    await index_changes(db_session, db_workspace.id)

    rows = await by_path(db_session, db_workspace.id)
    assert set(rows) == {"/documents/new.xml"}
    assert rows["/documents/new.xml"].id == document_id


# ── Convergence ─────────────────────────────────────────────────────────────


async def test_reindex_keeps_document_ids(
    store, db_session, db_workspace, patched_embed_texts
):
    """Rebuild replaces chunks, never document rows: their ids reach the browser."""
    await commit(store, {"documents/a.xml": "# A", "documents/notes/b.xml": "# B"})
    await index_changes(db_session, db_workspace.id)
    before = {t: d.id for t, d in (await titles(db_session, db_workspace.id)).items()}

    await index_tree(db_session, db_workspace.id)

    after = {t: d.id for t, d in (await titles(db_session, db_workspace.id)).items()}
    assert after == before


async def test_a_deleted_document_does_not_come_back_on_a_rebuild(
    store, db_session, db_workspace, workspace_flip, patched_embed_texts
):
    """The canary's bug, from the outside: the row went and the file stayed, so
    the rebuild read the file back as a document nobody asked for — and the
    drift check then agreed, both sides now holding the same resurrected copy."""
    workspace_flip(True)
    await commit(store, {"documents/a.xml": "# A"})
    await index_changes(db_session, db_workspace.id)
    document = (await titles(db_session, db_workspace.id))["a"]

    await record_deleted_documents(db_session, [document])
    await db_session.delete(document)
    await db_session.commit()

    await index_tree(db_session, db_workspace.id)

    assert await titles(db_session, db_workspace.id) == {}


async def test_reindex_reaches_the_same_state_as_the_incremental_path(
    store, db_session, db_workspace, patched_embed_texts
):
    await commit(store, {"documents/a.xml": "# A", "documents/b.xml": "# B"})
    await index_changes(db_session, db_workspace.id)
    await commit(
        store, {"documents/b.xml": "# B\n\nEdited."}, removes=["documents/a.xml"]
    )
    await index_changes(db_session, db_workspace.id)
    incremental = {
        title: document.source_markdown
        for title, document in (await titles(db_session, db_workspace.id)).items()
    }

    await index_tree(db_session, db_workspace.id)

    rebuilt = {
        title: document.source_markdown
        for title, document in (await titles(db_session, db_workspace.id)).items()
    }
    assert rebuilt == incremental


async def test_indexing_a_stamped_revision_does_nothing(
    store, db_session, db_workspace, patched_embed_texts
):
    await commit(store, {"documents/a.xml": "# A"})
    await index_changes(db_session, db_workspace.id)
    calls = patched_embed_texts.call_count

    outcome = await index_changes(db_session, db_workspace.id)

    assert outcome.indexed == 0
    assert patched_embed_texts.call_count == calls


async def test_a_rebuild_reembeds_nothing_when_the_tree_is_unchanged(
    store, db_session, db_workspace, patched_embed_texts
):
    """index_tree replays every path, not just what moved, so the hourly drift
    sweep would re-embed an unchanged workspace on every run. A READY row that
    still hashes to its body is left alone, keeping its chunk ids and the embedder
    idle."""
    await commit(store, {"documents/a.xml": "# A\n\nAlpha."})
    await index_tree(db_session, db_workspace.id)
    doc = (await titles(db_session, db_workspace.id))["a"]
    before = await chunk_ids(db_session, doc.id)
    assert before
    calls = patched_embed_texts.call_count

    await index_tree(db_session, db_workspace.id)

    assert await chunk_ids(db_session, doc.id) == before
    assert patched_embed_texts.call_count == calls


async def test_a_missed_revision_is_folded_in_by_the_next_run(
    store, db_session, db_workspace, patched_embed_texts
):
    """A dropped task must not strand its revision: the next run converges both."""
    await commit(store, {"documents/a.xml": "# A"})
    await commit(store, {"documents/b.xml": "# B"})  # no index run for this one

    await index_changes(db_session, db_workspace.id)

    assert set(await titles(db_session, db_workspace.id)) == {"a", "b"}
    await db_session.refresh(db_workspace)
    assert db_workspace.last_indexed_revision == await store.get_current_revision()


async def test_a_connector_document_survives_a_rebuild(
    store, db_session, db_workspace, db_user, patched_embed_texts
):
    """Prune is keyed on the ownership marker; connector rows have no path at all."""
    connector_row = Document(
        title="Slack thread",
        document_type=DocumentType.SLACK_CONNECTOR,
        document_metadata={"channel": "general"},
        content="Hello",
        content_hash=f"hash-{uuid.uuid4().hex}",
        unique_identifier_hash=f"unique-{uuid.uuid4().hex}",
        source_markdown="Hello",
        workspace_id=db_workspace.id,
        created_by_id=db_user.id,
        status=DocumentStatus.ready(),
    )
    db_session.add(connector_row)
    await db_session.flush()

    await commit(store, {"documents/a.xml": "# A"})
    await index_tree(db_session, db_workspace.id)

    assert "Slack thread" in await titles(db_session, db_workspace.id)


async def test_the_incremental_path_materializes_an_empty_keep_folder(
    store, db_session, db_workspace, patched_embed_texts
):
    """An agent's empty folder is a bare ``.keep`` the change loop skips as blank;
    the incremental path must still reconcile it into a row, not wait for a
    rebuild — the drift the two paths otherwise had."""
    async with store.transaction(message="mkdir", author=user_identity("1")) as tx:
        tx.write("documents/Smoking rules/.keep", b"")
    revision = tx.revision

    await index_changes(db_session, db_workspace.id)

    names = (
        (
            await db_session.execute(
                select(Folder.name).where(Folder.workspace_id == db_workspace.id)
            )
        )
        .scalars()
        .all()
    )
    assert "Smoking rules" in names
    assert await titles(db_session, db_workspace.id) == {}
    await db_session.refresh(db_workspace)
    assert db_workspace.last_indexed_revision == revision


async def test_a_rebuild_prunes_a_row_whose_file_is_gone(
    store, db_session, db_workspace, patched_embed_texts
):
    """The rebuild path has no change list, so the tree is the only authority."""
    await commit(store, {"documents/a.xml": "# A", "documents/b.xml": "# B"})
    await index_changes(db_session, db_workspace.id)

    # Remove the file without letting the incremental path see the removal.
    await commit(store, removes=["documents/a.xml"])
    await index_tree(db_session, db_workspace.id)

    assert set(await titles(db_session, db_workspace.id)) == {"b"}


async def test_a_rebuild_prunes_a_path_only_row_whose_file_is_gone(
    store, db_session, db_workspace, patched_embed_texts
):
    """A backfilled legacy row is identified by its ``path`` column, not a marker.
    Ownership — and so prune — has to key on the column, or such a row is never
    reclaimed when its file leaves the tree."""
    await commit(store, {"documents/a.xml": "# A", "documents/b.xml": "# B"})
    await index_changes(db_session, db_workspace.id)

    # A backfilled legacy row: the path column names its file, no marker.
    doc = (await titles(db_session, db_workspace.id))["a"]
    assert doc.path == "/documents/a.xml"
    doc.document_metadata = {}
    await db_session.commit()

    await commit(store, removes=["documents/a.xml"])
    await index_tree(db_session, db_workspace.id)

    assert set(await titles(db_session, db_workspace.id)) == {"b"}


# ── Authorship, skips, failures ─────────────────────────────────────────────


async def test_an_agent_authored_revision_falls_back_to_the_workspace_owner(
    store, db_session, db_workspace, db_user, patched_embed_texts
):
    """Autonomous writes carry no user id, and created_by_id rejects blanks."""
    await commit(
        store, {"documents/agent.xml": "# Written by the agent"}, author=AGENT_IDENTITY
    )

    await index_changes(db_session, db_workspace.id)

    assert (await titles(db_session, db_workspace.id))["agent"].created_by_id == (
        db_user.id
    )


async def test_unusable_blobs_are_skipped_and_the_stamp_still_advances(
    store, db_session, db_workspace, patched_embed_texts
):
    """A touched file and a binary are legal git; neither may wedge the workspace."""
    async with store.transaction(message="mixed", author=user_identity("1")) as tx:
        tx.write("documents/good.xml", b"# Good")
        tx.write("documents/blank.xml", b"   \n\n")
        tx.write("documents/binary.xml", b"\xff\xfe\x00\x01")
    revision = tx.revision

    outcome = await index_changes(db_session, db_workspace.id)

    assert set(await titles(db_session, db_workspace.id)) == {"good"}
    assert (outcome.indexed, outcome.skipped) == (1, 2)
    await db_session.refresh(db_workspace)
    assert db_workspace.last_indexed_revision == revision


async def test_the_git_path_chunks_identically_to_the_connector_path(
    store, db_session, db_workspace, db_user, patched_embed_texts
):
    """Differential, not a golden baseline: a baseline rots on the first re-chunk.

    This is also the only assertion that the shared chunk/cache/reconcile chain —
    which ships to every workspace, flag or no flag — still produces the same
    chunks and the same line spans it did for connectors.
    """
    markdown = (
        "# Report\n"
        "\n"
        "First paragraph with some detail.\n"
        "\n"
        "| col | val |\n"
        "| --- | --- |\n"
        "| a   | 1   |\n"
        "\n"
        "Closing paragraph.\n"
    )
    through_connector = Document(
        title="Report via connector",
        document_type=DocumentType.SLACK_CONNECTOR,
        document_metadata={},
        content=markdown,
        content_hash=f"hash-{uuid.uuid4().hex}",
        unique_identifier_hash=f"unique-{uuid.uuid4().hex}",
        source_markdown=markdown,
        workspace_id=db_workspace.id,
        created_by_id=db_user.id,
        status=DocumentStatus.pending(),
    )
    db_session.add(through_connector)
    await db_session.flush()
    await IndexingPipelineService(db_session).index(
        through_connector,
        ConnectorDocument(
            title=through_connector.title,
            source_markdown=markdown,
            unique_id="slack-1",
            document_type=DocumentType.SLACK_CONNECTOR,
            workspace_id=db_workspace.id,
            created_by_id=str(db_user.id),
        ),
    )

    await commit(store, {"documents/Report.xml": markdown})
    await index_changes(db_session, db_workspace.id)
    through_git = (await titles(db_session, db_workspace.id))["Report"]

    assert await chunk_shapes(db_session, through_git.id) == await chunk_shapes(
        db_session, through_connector.id
    )


async def test_a_failed_document_withholds_the_stamp(
    store, db_session, db_workspace, patched_embed_texts_raises
):
    """Without this the sweep has nothing to retry and the gap is permanent."""
    await commit(store, {"documents/a.xml": "# A"})

    outcome = await index_changes(db_session, db_workspace.id)

    assert outcome.failed == 1
    assert outcome.stamped is False
    await db_session.refresh(db_workspace)
    assert db_workspace.last_indexed_revision is None


async def test_a_failing_document_does_not_discard_its_batch(
    store, db_session, db_workspace, monkeypatch
):
    """One un-indexable file must not roll back the rows of the documents that
    indexed cleanly beside it in the same run."""
    from app.indexing_pipeline.cache import cached_indexing

    real_embed = cached_indexing.embed_texts

    def selective(texts):
        if any("POISON" in text for text in texts):
            raise RuntimeError("embedding unavailable")
        return real_embed(texts)

    monkeypatch.setattr(cached_indexing, "embed_texts", selective)

    await commit(
        store,
        {
            "documents/good.xml": "# Good\n\na clean body to index\n",
            "documents/bad.xml": "# Bad\n\nPOISON body that cannot embed\n",
        },
    )

    workspace_id = db_workspace.id
    outcome = await index_changes(db_session, workspace_id)

    assert outcome.failed == 1
    db_session.expire_all()
    good = await db_session.scalar(
        select(Document).where(
            Document.workspace_id == workspace_id,
            Document.path == "/documents/good.xml",
        )
    )
    assert good is not None
    assert DocumentStatus.is_state(good.status, DocumentStatus.READY)
    chunk_count = await db_session.scalar(
        select(func.count(Chunk.id)).where(Chunk.document_id == good.id)
    )
    assert chunk_count and chunk_count > 0


async def test_circleback_type_survives_rebuild_after_row_loss(
    store,
    db_session,
    db_workspace,
    db_user,
    workspace_flip,
    patched_embed_texts,
):
    """Restore Circleback identity when its PostgreSQL row is lost."""
    from app.knowledge_store.service import record_prepared_documents

    workspace_flip(True)

    meeting_id = "WRweb12iCBimU6Vo8cD7z"
    markdown = "# Release Review\\n\\n**Alice** [00:03]: Approved.\\n"

    original = Document(
        title="Release Review",
        document_type=DocumentType.CIRCLEBACK,
        document_metadata={
            "CIRCLEBACK_MEETING_ID": meeting_id,
            "SOURCE": "CIRCLEBACK_WEBHOOK",
        },
        content=markdown,
        source_markdown=markdown,
        content_hash=f"test-{uuid.uuid4().hex}",
        unique_identifier_hash=generate_unique_identifier_hash(
            DocumentType.CIRCLEBACK,
            f"circleback_{meeting_id}",
            db_workspace.id,
        ),
        workspace_id=db_workspace.id,
        created_by_id=db_user.id,
        status=DocumentStatus.ready(),
    )

    db_session.add(original)
    await db_session.commit()

    revision = await record_prepared_documents(db_session, [original])
    assert revision is not None

    original_path = original.path
    assert original_path is not None

    # Simulate loss of the database row without deleting the Git file.
    await db_session.delete(original)
    await db_session.commit()

    await index_tree(db_session, db_workspace.id)

    restored = await db_session.scalar(
        select(Document).where(
            Document.workspace_id == db_workspace.id,
            Document.path == original_path,
        )
    )

    assert restored is not None
    assert restored.document_type == DocumentType.CIRCLEBACK
    assert restored.document_metadata["CIRCLEBACK_MEETING_ID"] == meeting_id


async def test_circleback_ingest_commits_markdown_and_identity_together(
    store, db_session, db_workspace, db_user, workspace_flip
):
    """Circleback content and identity must share a Git revision."""
    from app.knowledge_store.circleback_identity import (
        decode_identity,
        identity_path,
    )
    from app.knowledge_store.paths import to_store_path
    from app.knowledge_store.service import record_prepared_documents

    workspace_flip(True)

    meeting_id = "WRweb12iCBimU6Vo8cD7z"
    markdown = "# Release Review\n\n**Alice** [00:03]: Approved.\n"

    document = Document(
        title="Release Review",
        document_type=DocumentType.CIRCLEBACK,
        document_metadata={"CIRCLEBACK_MEETING_ID": meeting_id},
        content=markdown,
        source_markdown=markdown,
        content_hash=f"test-{uuid.uuid4().hex}",
        unique_identifier_hash=generate_unique_identifier_hash(
            DocumentType.CIRCLEBACK,
            f"circleback_{meeting_id}",
            db_workspace.id,
        ),
        workspace_id=db_workspace.id,
        created_by_id=db_user.id,
        status=DocumentStatus.ready(),
    )

    db_session.add(document)
    await db_session.commit()

    revision = await record_prepared_documents(db_session, [document])
    assert revision is not None
    assert document.path is not None

    markdown_path = to_store_path(document.path)
    metadata_path = identity_path(document.path)

    tracked = {entry.path for entry in await store.list_paths(revision)}
    assert markdown_path in tracked
    assert metadata_path in tracked

    saved_markdown = await store.read_as_of(revision, markdown_path)
    saved_identity = await store.read_as_of(revision, metadata_path)

    assert saved_markdown.decode() == markdown
    assert decode_identity(
        saved_identity,
        virtual_path=document.path,
        markdown=markdown,
    ) == {
        "CIRCLEBACK_MEETING_ID": meeting_id,
        "SOURCE": "CIRCLEBACK_WEBHOOK",
    }


async def test_deleting_circleback_removes_its_git_identity(
    store, db_session, db_workspace, db_user, workspace_flip
):
    """Deleting a meeting must remove both content and identity."""
    from app.knowledge_store.circleback_identity import identity_path
    from app.knowledge_store.paths import to_store_path
    from app.knowledge_store.service import (
        record_deleted_documents,
        record_prepared_documents,
    )

    workspace_flip(True)
    meeting_id = "delete-test-meeting-001"
    markdown = "# Deletion test\n\n**Alice** [00:03]: Done.\n"

    document = Document(
        title="Deletion test",
        document_type=DocumentType.CIRCLEBACK,
        document_metadata={"CIRCLEBACK_MEETING_ID": meeting_id},
        content=markdown,
        source_markdown=markdown,
        content_hash=f"test-{uuid.uuid4().hex}",
        unique_identifier_hash=generate_unique_identifier_hash(
            DocumentType.CIRCLEBACK,
            f"circleback_{meeting_id}",
            db_workspace.id,
        ),
        workspace_id=db_workspace.id,
        created_by_id=db_user.id,
        status=DocumentStatus.ready(),
    )

    db_session.add(document)
    await db_session.commit()

    created_revision = await record_prepared_documents(db_session, [document])
    assert created_revision is not None
    assert document.path is not None

    markdown_path = to_store_path(document.path)
    metadata_path = identity_path(document.path)

    before = {entry.path for entry in await store.list_paths(created_revision)}
    assert markdown_path in before
    assert metadata_path in before

    deleted_revision = await record_deleted_documents(db_session, [document])
    assert deleted_revision is not None

    after = {entry.path for entry in await store.list_paths(deleted_revision)}

    assert markdown_path not in after
    assert metadata_path not in after


async def test_moving_circleback_relocates_git_identity(
    store, db_session, db_workspace, db_user, workspace_flip
):
    """A meeting rename must relocate its matching Git identity."""
    from app.knowledge_store.circleback_identity import (
        decode_identity,
        identity_path,
    )
    from app.knowledge_store.paths import to_store_path
    from app.knowledge_store.service import (
        record_moved_documents,
        record_prepared_documents,
    )

    workspace_flip(True)
    meeting_id = "move-test-meeting-001"
    markdown = "# Release Review\\n\\n**Alice** [00:03]: Approved.\\n"

    document = Document(
        title="Release Review",
        document_type=DocumentType.CIRCLEBACK,
        document_metadata={"CIRCLEBACK_MEETING_ID": meeting_id},
        content=markdown,
        source_markdown=markdown,
        content_hash=f"test-{uuid.uuid4().hex}",
        unique_identifier_hash=generate_unique_identifier_hash(
            DocumentType.CIRCLEBACK,
            f"circleback_{meeting_id}",
            db_workspace.id,
        ),
        workspace_id=db_workspace.id,
        created_by_id=db_user.id,
        status=DocumentStatus.ready(),
    )

    db_session.add(document)
    await db_session.commit()

    original_revision = await record_prepared_documents(db_session, [document])
    assert original_revision is not None

    old_path = document.path
    assert old_path is not None
    old_identity_path = identity_path(old_path)

    document.title = "Release Review Final"
    revision = await record_moved_documents(db_session, [document])

    assert revision is not None
    new_path = document.path
    assert new_path is not None
    assert new_path != old_path

    new_identity_path = identity_path(new_path)

    tracked = {entry.path for entry in await store.list_paths(revision)}

    assert to_store_path(old_path) not in tracked
    assert to_store_path(new_path) in tracked

    assert old_identity_path not in tracked
    assert new_identity_path in tracked

    saved_markdown = await store.read_as_of(revision, to_store_path(new_path))
    saved_identity = await store.read_as_of(revision, new_identity_path)

    assert saved_markdown.decode() == markdown
    assert decode_identity(
        saved_identity,
        virtual_path=new_path,
        markdown=markdown,
    ) == {
        "CIRCLEBACK_MEETING_ID": meeting_id,
        "SOURCE": "CIRCLEBACK_WEBHOOK",
    }


async def test_editor_retitle_updates_circleback_git_identity(
    store, db_session, db_workspace, db_user, workspace_flip
):
    """Editor rename and content change must refresh Circleback identity."""
    from app.knowledge_store.circleback_identity import (
        decode_identity,
        identity_path,
    )
    from app.knowledge_store.paths import to_store_path
    from app.knowledge_store.service import (
        record_prepared_documents,
        record_saved_document,
    )

    workspace_flip(True)

    meeting_id = "editor-test-meeting-001"
    original = "# Release Review\\n\\n**Alice** [00:03]: Draft.\\n"
    updated = "# Release Review Final\\n\\n**Alice** [00:03]: Approved.\\n"

    document = Document(
        title="Release Review",
        document_type=DocumentType.CIRCLEBACK,
        document_metadata={"CIRCLEBACK_MEETING_ID": meeting_id},
        content=original,
        source_markdown=original,
        content_hash=f"test-{uuid.uuid4().hex}",
        unique_identifier_hash=generate_unique_identifier_hash(
            DocumentType.CIRCLEBACK,
            f"circleback_{meeting_id}",
            db_workspace.id,
        ),
        workspace_id=db_workspace.id,
        created_by_id=db_user.id,
        status=DocumentStatus.ready(),
    )

    db_session.add(document)
    await db_session.commit()

    first_revision = await record_prepared_documents(db_session, [document])
    assert first_revision is not None

    old_path = document.path
    assert old_path is not None

    document.title = "Release Review Final"
    revision = await record_saved_document(
        db_session,
        workspace_id=db_workspace.id,
        doc_id=document.id,
        title="Release Review Final",
        folder_id=document.folder_id,
        markdown=updated,
        author_user_id=str(db_user.id),
        title_is_explicit=True,
    )
    assert revision is not None

    new_path = document.path
    assert new_path is not None
    assert new_path != old_path

    tracked = {entry.path for entry in await store.list_paths(revision)}

    assert to_store_path(old_path) not in tracked
    assert to_store_path(new_path) in tracked
    assert identity_path(old_path) not in tracked
    assert identity_path(new_path) in tracked

    saved = await store.read_as_of(revision, to_store_path(new_path))
    assert saved.decode() == updated

    raw_identity = await store.read_as_of(revision, identity_path(new_path))
    assert decode_identity(
        raw_identity,
        virtual_path=new_path,
        markdown=updated,
    ) == {
        "CIRCLEBACK_MEETING_ID": meeting_id,
        "SOURCE": "CIRCLEBACK_WEBHOOK",
    }


async def test_editor_body_only_update_preserves_circleback_identity(
    store,
    db_session,
    db_workspace,
    db_user,
    workspace_flip,
    patched_embed_texts,
):
    """A body-only edit refreshes identity without moving the document."""
    from app.knowledge_store.circleback_identity import (
        decode_identity,
        identity_path,
    )
    from app.knowledge_store.paths import to_store_path
    from app.knowledge_store.service import (
        record_prepared_documents,
        record_saved_document,
    )

    workspace_flip(True)
    meeting_id = "body-only-test-001"
    original = "# Release Review\\n\\n**Alice** [00:03]: Draft.\\n"
    updated = "# Release Review\\n\\n**Alice** [00:03]: Approved.\\n"

    document = Document(
        title="Release Review",
        document_type=DocumentType.CIRCLEBACK,
        document_metadata={"CIRCLEBACK_MEETING_ID": meeting_id},
        content=original,
        source_markdown=original,
        content_hash=f"test-{uuid.uuid4().hex}",
        unique_identifier_hash=generate_unique_identifier_hash(
            DocumentType.CIRCLEBACK,
            f"circleback_{meeting_id}",
            db_workspace.id,
        ),
        workspace_id=db_workspace.id,
        created_by_id=db_user.id,
        status=DocumentStatus.ready(),
    )
    db_session.add(document)
    await db_session.commit()

    first_revision = await record_prepared_documents(db_session, [document])
    assert first_revision is not None

    original_path = document.path
    assert original_path is not None
    sidecar_path = identity_path(original_path)

    previous_identity = await store.read_as_of(first_revision, sidecar_path)

    revision = await record_saved_document(
        db_session,
        workspace_id=db_workspace.id,
        doc_id=document.id,
        title=document.title,
        folder_id=document.folder_id,
        markdown=updated,
        author_user_id=str(db_user.id),
        title_is_explicit=False,
    )

    assert revision is not None
    assert document.path == original_path

    paths = {item.path for item in await store.list_paths(revision)}
    assert to_store_path(original_path) in paths
    assert sidecar_path in paths

    saved_markdown = await store.read_as_of(revision, to_store_path(original_path))
    saved_identity = await store.read_as_of(revision, sidecar_path)

    assert saved_markdown.decode() == updated
    assert saved_identity != previous_identity
    assert decode_identity(
        saved_identity,
        virtual_path=original_path,
        markdown=updated,
    ) == {
        "CIRCLEBACK_MEETING_ID": meeting_id,
        "SOURCE": "CIRCLEBACK_WEBHOOK",
    }

    # Confirm that the updated Git evidence can rebuild a lost DB row.
    await db_session.delete(document)
    await db_session.commit()

    await index_tree(db_session, db_workspace.id)

    restored = await db_session.scalar(
        select(Document).where(
            Document.workspace_id == db_workspace.id,
            Document.path == original_path,
        )
    )

    assert restored is not None
    assert restored.document_type == DocumentType.CIRCLEBACK
    assert restored.source_markdown == updated
    assert restored.document_metadata["CIRCLEBACK_MEETING_ID"] == meeting_id
    assert restored.unique_identifier_hash == (
        generate_unique_identifier_hash(
            DocumentType.CIRCLEBACK,
            f"circleback_{meeting_id}",
            db_workspace.id,
        )
    )


async def test_moving_folder_relocates_circleback_identity(
    store, db_session, db_workspace, db_user, workspace_flip
):
    """Moving a parent folder must update its Circleback identity."""
    from app.knowledge_store.circleback_identity import (
        decode_identity,
        identity_path,
    )
    from app.knowledge_store.paths import to_store_path
    from app.knowledge_store.service import record_prepared_documents
    from app.services.folder_service import ensure_folder_hierarchy

    workspace_flip(True)

    folder_id = await ensure_folder_hierarchy(
        db_session,
        workspace_id=db_workspace.id,
        created_by_id=str(db_user.id),
        folder_parts=["Meetings"],
    )

    meeting_id = "folder-move-test-001"
    markdown = "# Review\\n\\n**Alice** [00:03]: Approved.\\n"

    document = Document(
        title="Review",
        document_type=DocumentType.CIRCLEBACK,
        document_metadata={"CIRCLEBACK_MEETING_ID": meeting_id},
        content=markdown,
        source_markdown=markdown,
        content_hash=f"test-{uuid.uuid4().hex}",
        unique_identifier_hash=generate_unique_identifier_hash(
            DocumentType.CIRCLEBACK,
            f"circleback_{meeting_id}",
            db_workspace.id,
        ),
        workspace_id=db_workspace.id,
        folder_id=folder_id,
        created_by_id=db_user.id,
        status=DocumentStatus.ready(),
    )
    db_session.add(document)
    await db_session.commit()

    first = await record_prepared_documents(db_session, [document])
    assert first is not None

    old_path = document.path
    assert old_path is not None
    assert old_path.startswith("/documents/Meetings/")

    bound_store = store.with_session(db_session).as_user(str(db_user.id))
    outcome = await bound_store.move_folder(
        "/documents/Meetings",
        "/documents/Archive",
    )

    assert outcome.revision is not None
    new_path = old_path.replace(
        "/documents/Meetings/",
        "/documents/Archive/",
        1,
    )

    tracked = {entry.path for entry in await store.list_paths(outcome.revision)}

    assert to_store_path(old_path) not in tracked
    assert to_store_path(new_path) in tracked

    assert identity_path(old_path) not in tracked
    assert identity_path(new_path) in tracked

    raw_identity = await store.read_as_of(
        outcome.revision,
        identity_path(new_path),
    )

    assert decode_identity(
        raw_identity,
        virtual_path=new_path,
        markdown=markdown,
    ) == {
        "CIRCLEBACK_MEETING_ID": meeting_id,
        "SOURCE": "CIRCLEBACK_WEBHOOK",
    }


async def test_removing_folder_cleans_circleback_identities(
    store, db_session, db_workspace, db_user, workspace_flip
):
    """Deleting a folder removes meeting content and identity sidecars."""
    from app.knowledge_store.circleback_identity import identity_path
    from app.knowledge_store.paths import to_store_path
    from app.knowledge_store.service import record_prepared_documents
    from app.services.folder_service import ensure_folder_hierarchy

    workspace_flip(True)

    folder_id = await ensure_folder_hierarchy(
        db_session,
        workspace_id=db_workspace.id,
        created_by_id=str(db_user.id),
        folder_parts=["Old Meetings"],
    )

    meeting_id = "folder-delete-test-001"
    markdown = "# Old Meeting\\n\\n**Alice** [00:03]: Done.\\n"

    document = Document(
        title="Old Meeting",
        document_type=DocumentType.CIRCLEBACK,
        document_metadata={"CIRCLEBACK_MEETING_ID": meeting_id},
        content=markdown,
        source_markdown=markdown,
        content_hash=f"test-{uuid.uuid4().hex}",
        unique_identifier_hash=generate_unique_identifier_hash(
            DocumentType.CIRCLEBACK,
            f"circleback_{meeting_id}",
            db_workspace.id,
        ),
        workspace_id=db_workspace.id,
        folder_id=folder_id,
        created_by_id=db_user.id,
        status=DocumentStatus.ready(),
    )
    db_session.add(document)
    await db_session.commit()

    first = await record_prepared_documents(db_session, [document])
    assert first is not None

    original_path = document.path
    assert original_path is not None

    content_path = to_store_path(original_path)
    sidecar_path = identity_path(original_path)

    before = {item.path for item in await store.list_paths(first)}
    assert content_path in before
    assert sidecar_path in before

    bound_store = store.with_session(db_session).as_user(str(db_user.id))
    outcome = await bound_store.remove_folder("/documents/Old Meetings")

    assert outcome.revision is not None

    after = {item.path for item in await store.list_paths(outcome.revision)}

    assert content_path not in after
    assert sidecar_path not in after


async def test_mixed_folder_batch_preserves_and_cleans_meeting_identities(
    store, db_session, db_workspace, db_user, workspace_flip
):
    """Two meetings and a note survive batch move without identity mixups."""
    from app.knowledge_store.circleback_identity import (
        decode_identity,
        identity_path,
    )
    from app.knowledge_store.paths import to_store_path
    from app.knowledge_store.service import record_prepared_documents
    from app.services.folder_service import ensure_folder_hierarchy

    workspace_flip(True)

    folder_id = await ensure_folder_hierarchy(
        db_session,
        workspace_id=db_workspace.id,
        created_by_id=str(db_user.id),
        folder_parts=["Team Meetings"],
    )

    definitions = [
        ("Sprint Planning", "mixed-meeting-001", "# Sprint Planning\\nA\\n"),
        ("Design Review", "mixed-meeting-002", "# Design Review\\nB\\n"),
        ("General Note", None, "# General Note\\nC\\n"),
    ]
    documents = []

    for title, meeting_id, markdown in definitions:
        kind = DocumentType.CIRCLEBACK if meeting_id is not None else DocumentType.NOTE
        document = Document(
            title=title,
            document_type=kind,
            document_metadata=(
                {"CIRCLEBACK_MEETING_ID": meeting_id} if meeting_id is not None else {}
            ),
            content=markdown,
            source_markdown=markdown,
            content_hash=f"batch-{uuid.uuid4().hex}",
            unique_identifier_hash=(
                generate_unique_identifier_hash(
                    DocumentType.CIRCLEBACK,
                    f"circleback_{meeting_id}",
                    db_workspace.id,
                )
                if meeting_id is not None
                else f"batch-note-{uuid.uuid4().hex}"
            ),
            workspace_id=db_workspace.id,
            folder_id=folder_id,
            created_by_id=db_user.id,
            status=DocumentStatus.ready(),
        )
        db_session.add(document)
        documents.append(document)

    await db_session.commit()

    first = await record_prepared_documents(db_session, documents)
    assert first is not None

    original_paths = [doc.path for doc in documents]
    assert all(original_paths)
    assert len(set(original_paths)) == 3

    bound_store = store.with_session(db_session).as_user(str(db_user.id))
    moved = await bound_store.move_folder(
        "/documents/Team Meetings",
        "/documents/Archive",
    )
    assert moved.revision is not None

    moved_paths = {item.path for item in await store.list_paths(moved.revision)}

    new_paths = []

    for (_title, meeting_id, markdown), old_path in zip(
        definitions, original_paths, strict=True
    ):
        new_path = old_path.replace(
            "/documents/Team Meetings/",
            "/documents/Archive/",
            1,
        )
        assert new_path != old_path
        new_paths.append(new_path)

        assert to_store_path(old_path) not in moved_paths
        assert to_store_path(new_path) in moved_paths

        saved = await store.read_as_of(moved.revision, to_store_path(new_path))
        assert saved.decode() == markdown

        if meeting_id is None:
            assert identity_path(old_path) not in moved_paths
            assert identity_path(new_path) not in moved_paths
        else:
            assert identity_path(old_path) not in moved_paths
            assert identity_path(new_path) in moved_paths

            raw = await store.read_as_of(moved.revision, identity_path(new_path))
            assert (
                decode_identity(
                    raw,
                    virtual_path=new_path,
                    markdown=markdown,
                )["CIRCLEBACK_MEETING_ID"]
                == meeting_id
            )

    deleted = await bound_store.remove_folder("/documents/Archive")
    assert deleted.revision is not None

    remaining = {item.path for item in await store.list_paths(deleted.revision)}

    for new_path in new_paths:
        assert to_store_path(new_path) not in remaining
        assert identity_path(new_path) not in remaining

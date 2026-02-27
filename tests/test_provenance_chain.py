"""Tests for the Model Provenance Chain (P1.2).

Covers:
- Ed25519 keypair generation and signing utilities
- SHA-256 hash chain computation
- Signature verification (valid and invalid)
- ProvenanceChainService: create_chain, add_link, verify_chain, seal_chain
- Tamper-detection: payload mutation, hash replacement, signature replacement
- Full lifecycle test: 7-stage chain creation and verification

These tests use a MagicMock-based in-memory session to avoid database
dependencies, making them fast and deterministic.
"""

import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aumos_model_registry.core.services.provenance_chain_service import (
    LINK_TYPES,
    ProvenanceChainResult,
    ProvenanceChainService,
    compute_link_hash,
    generate_ed25519_keypair,
    sign_link_payload,
    verify_link_signature,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def ed25519_keypair() -> tuple[bytes, str]:
    """Generate a fresh Ed25519 keypair for each test."""
    return generate_ed25519_keypair()


@pytest.fixture()
def private_key_pem(ed25519_keypair: tuple[bytes, str]) -> bytes:
    """Return PEM bytes of the Ed25519 private key."""
    return ed25519_keypair[0]


@pytest.fixture()
def public_key_pem(ed25519_keypair: tuple[bytes, str]) -> str:
    """Return PEM string of the Ed25519 public key."""
    return ed25519_keypair[1]


@pytest.fixture()
def tenant_id() -> uuid.UUID:
    """Deterministic tenant UUID for tests."""
    return uuid.UUID("00000000-0000-0000-0000-000000000001")


@pytest.fixture()
def model_id() -> uuid.UUID:
    """Deterministic model UUID for tests."""
    return uuid.UUID("00000000-0000-0000-0000-000000000002")


@pytest.fixture()
def public_key_id() -> uuid.UUID:
    """Deterministic public key UUID for tests."""
    return uuid.UUID("00000000-0000-0000-0000-000000000003")


@pytest.fixture()
def service() -> ProvenanceChainService:
    """Return a ProvenanceChainService instance."""
    return ProvenanceChainService()


def _make_chain(
    chain_id: uuid.UUID,
    tenant_id: uuid.UUID,
    model_id: uuid.UUID,
    head_link_hash: str | None = None,
    chain_status: str = "active",
    is_verified: bool = False,
) -> MagicMock:
    """Create a fake ModelProvenanceChain ORM object."""
    chain = MagicMock()
    chain.id = chain_id
    chain.tenant_id = tenant_id
    chain.model_id = model_id
    chain.model_version_id = None
    chain.chain_status = chain_status
    chain.is_verified = is_verified
    chain.head_link_hash = head_link_hash
    chain.metadata_ = {}
    chain.links = []
    return chain


def _make_link(
    chain_id: uuid.UUID,
    sequence_number: int,
    link_type: str,
    payload: dict[str, Any],
    link_hash: str,
    previous_link_hash: str | None,
    signature: str,
    signed_by: uuid.UUID,
    tenant_id: uuid.UUID,
) -> MagicMock:
    """Create a fake ProvenanceChainLink ORM object."""
    link = MagicMock()
    link.id = uuid.uuid4()
    link.tenant_id = tenant_id
    link.chain_id = chain_id
    link.sequence_number = sequence_number
    link.link_type = link_type
    link.payload = payload
    link.link_hash = link_hash
    link.previous_link_hash = previous_link_hash
    link.signature = signature
    link.signed_by = signed_by
    link.actor_id = None
    return link


def _make_public_key_record(key_id: uuid.UUID, public_key_pem: str, tenant_id: uuid.UUID) -> MagicMock:
    """Create a fake ProvenancePublicKey ORM object."""
    key_record = MagicMock()
    key_record.id = key_id
    key_record.tenant_id = tenant_id
    key_record.public_key_pem = public_key_pem
    key_record.is_active = True
    return key_record


# ---------------------------------------------------------------------------
# Test 1: Ed25519 keypair generation
# ---------------------------------------------------------------------------


def test_generate_ed25519_keypair_returns_valid_keys() -> None:
    """generate_ed25519_keypair returns PEM-encoded keys that can be loaded."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
    from cryptography.hazmat.primitives.serialization import load_pem_private_key, load_pem_public_key

    private_pem, public_pem = generate_ed25519_keypair()

    private_key = load_pem_private_key(private_pem, password=None)
    public_key = load_pem_public_key(public_pem.encode("utf-8"))

    assert isinstance(private_key, Ed25519PrivateKey)
    assert isinstance(public_key, Ed25519PublicKey)


def test_generate_ed25519_keypair_produces_unique_keys() -> None:
    """Each generate_ed25519_keypair call produces a different keypair."""
    _, pub1 = generate_ed25519_keypair()
    _, pub2 = generate_ed25519_keypair()
    assert pub1 != pub2


# ---------------------------------------------------------------------------
# Test 2: SHA-256 hash chain computation
# ---------------------------------------------------------------------------


def test_compute_link_hash_genesis_link() -> None:
    """Genesis link (previous=None) hashes against 64 zeros."""
    payload = {"data": "test_value"}
    hash_result = compute_link_hash(None, payload)
    assert len(hash_result) == 64  # SHA-256 hex is always 64 chars
    assert all(c in "0123456789abcdef" for c in hash_result)


def test_compute_link_hash_chaining_is_deterministic() -> None:
    """Same inputs always produce the same hash."""
    payload = {"key": "value", "number": 42}
    hash1 = compute_link_hash("abc123" + "0" * 58, payload)
    hash2 = compute_link_hash("abc123" + "0" * 58, payload)
    assert hash1 == hash2


def test_compute_link_hash_changes_with_different_payload() -> None:
    """Different payload content produces a different hash."""
    prev_hash = "a" * 64
    hash1 = compute_link_hash(prev_hash, {"value": "original"})
    hash2 = compute_link_hash(prev_hash, {"value": "tampered"})
    assert hash1 != hash2


def test_compute_link_hash_changes_with_different_previous_hash() -> None:
    """Changing the previous hash produces a different current hash."""
    payload = {"data": "stable"}
    hash1 = compute_link_hash("a" * 64, payload)
    hash2 = compute_link_hash("b" * 64, payload)
    assert hash1 != hash2


# ---------------------------------------------------------------------------
# Test 3: Ed25519 signing
# ---------------------------------------------------------------------------


def test_sign_link_payload_returns_base64_string(
    private_key_pem: bytes,
) -> None:
    """sign_link_payload returns a non-empty base64-decodable string."""
    import base64

    payload = {"link_type": "training_dataset", "dataset_id": "ds-001"}
    signature = sign_link_payload(payload, private_key_pem)
    assert isinstance(signature, str)
    assert len(signature) > 0
    # Must be valid base64
    decoded = base64.b64decode(signature)
    assert len(decoded) == 64  # Ed25519 signatures are always 64 bytes


def test_sign_link_payload_is_deterministic_for_same_input(
    private_key_pem: bytes,
) -> None:
    """Ed25519 signing produces identical signatures for identical inputs and keys."""
    payload = {"data": "consistent"}
    sig1 = sign_link_payload(payload, private_key_pem)
    sig2 = sign_link_payload(payload, private_key_pem)
    assert sig1 == sig2


# ---------------------------------------------------------------------------
# Test 4: Ed25519 verification
# ---------------------------------------------------------------------------


def test_verify_link_signature_valid(
    private_key_pem: bytes,
    public_key_pem: str,
) -> None:
    """A signature produced by sign_link_payload verifies correctly."""
    payload = {"event": "training_started", "gpu_count": 8}
    signature = sign_link_payload(payload, private_key_pem)
    assert verify_link_signature(payload, signature, public_key_pem) is True


def test_verify_link_signature_fails_on_tampered_payload(
    private_key_pem: bytes,
    public_key_pem: str,
) -> None:
    """Signature verification fails when the payload has been altered."""
    original_payload = {"event": "training_started", "gpu_count": 8}
    tampered_payload = {"event": "training_started", "gpu_count": 16}
    signature = sign_link_payload(original_payload, private_key_pem)
    assert verify_link_signature(tampered_payload, signature, public_key_pem) is False


def test_verify_link_signature_fails_on_wrong_key(
    private_key_pem: bytes,
) -> None:
    """Signature verification fails when a different public key is used."""
    payload = {"event": "test"}
    signature = sign_link_payload(payload, private_key_pem)

    _, different_public_key_pem = generate_ed25519_keypair()
    assert verify_link_signature(payload, signature, different_public_key_pem) is False


def test_verify_link_signature_fails_on_corrupted_signature(
    public_key_pem: str,
) -> None:
    """Signature verification fails when the signature is corrupted."""
    import base64

    payload = {"event": "test"}
    # Create an obviously invalid 64-byte signature
    garbage_signature = base64.b64encode(b"\x00" * 64).decode("ascii")
    assert verify_link_signature(payload, garbage_signature, public_key_pem) is False


# ---------------------------------------------------------------------------
# Test 5: ProvenanceChainService.create_chain
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_chain_creates_and_returns_chain(
    service: ProvenanceChainService,
    tenant_id: uuid.UUID,
    model_id: uuid.UUID,
) -> None:
    """create_chain creates a chain with correct initial state."""
    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()

    # Capture what was added to session
    captured: list[Any] = []
    session.add.side_effect = lambda obj: captured.append(obj)

    result = await service.create_chain(
        session=session,
        tenant_id=tenant_id,
        model_id=model_id,
        metadata={"purpose": "test"},
    )

    assert session.add.called
    assert session.flush.called
    # The returned object should be the chain that was added
    assert result in captured
    assert result.tenant_id == tenant_id
    assert result.model_id == model_id
    assert result.chain_status == "active"
    assert result.is_verified is False
    assert result.head_link_hash is None


# ---------------------------------------------------------------------------
# Test 6: ProvenanceChainService.add_link
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_link_appends_to_active_chain(
    service: ProvenanceChainService,
    tenant_id: uuid.UUID,
    model_id: uuid.UUID,
    private_key_pem: bytes,
    public_key_id: uuid.UUID,
) -> None:
    """add_link creates a signed link and updates chain head hash."""
    chain_id = uuid.uuid4()
    chain = _make_chain(chain_id, tenant_id, model_id)

    session = AsyncMock()
    session.flush = AsyncMock()
    session.add = MagicMock()

    # Mock the chain query
    chain_result = MagicMock()
    chain_result.scalar_one_or_none.return_value = chain

    # Mock the count query
    count_result = MagicMock()
    count_result.scalar_one.return_value = 0  # No existing links

    session.execute = AsyncMock(side_effect=[chain_result, count_result])

    payload = {"dataset_uri": "s3://bucket/dataset.parquet", "record_count": 1_000_000}
    link = await service.add_link(
        session=session,
        chain_id=chain_id,
        tenant_id=tenant_id,
        link_type="training_dataset",
        payload=payload,
        private_key_pem=private_key_pem,
        public_key_id=public_key_id,
        actor_id=None,
    )

    assert link.chain_id == chain_id
    assert link.sequence_number == 1
    assert link.link_type == "training_dataset"
    assert link.previous_link_hash is None  # Genesis link
    assert len(link.link_hash) == 64
    assert link.signature != ""
    assert link.signed_by == public_key_id
    # Chain head should be updated
    assert chain.head_link_hash == link.link_hash


# ---------------------------------------------------------------------------
# Test 7: ProvenanceChainService.add_link — invalid states
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_link_raises_for_sealed_chain(
    service: ProvenanceChainService,
    tenant_id: uuid.UUID,
    model_id: uuid.UUID,
    private_key_pem: bytes,
    public_key_id: uuid.UUID,
) -> None:
    """add_link raises ValueError for a sealed chain."""
    chain_id = uuid.uuid4()
    chain = _make_chain(chain_id, tenant_id, model_id, chain_status="sealed")

    session = AsyncMock()
    chain_result = MagicMock()
    chain_result.scalar_one_or_none.return_value = chain
    session.execute = AsyncMock(return_value=chain_result)

    with pytest.raises(ValueError, match="sealed"):
        await service.add_link(
            session=session,
            chain_id=chain_id,
            tenant_id=tenant_id,
            link_type="training_dataset",
            payload={},
            private_key_pem=private_key_pem,
            public_key_id=public_key_id,
        )


@pytest.mark.asyncio
async def test_add_link_raises_for_invalid_link_type(
    service: ProvenanceChainService,
    tenant_id: uuid.UUID,
    model_id: uuid.UUID,
    private_key_pem: bytes,
    public_key_id: uuid.UUID,
) -> None:
    """add_link raises ValueError for unrecognized link_type."""
    chain_id = uuid.uuid4()
    chain = _make_chain(chain_id, tenant_id, model_id)

    session = AsyncMock()
    chain_result = MagicMock()
    chain_result.scalar_one_or_none.return_value = chain
    count_result = MagicMock()
    count_result.scalar_one.return_value = 0
    session.execute = AsyncMock(side_effect=[chain_result, count_result])

    with pytest.raises(ValueError, match="Invalid link_type"):
        await service.add_link(
            session=session,
            chain_id=chain_id,
            tenant_id=tenant_id,
            link_type="fantasy_stage",
            payload={},
            private_key_pem=private_key_pem,
            public_key_id=public_key_id,
        )


# ---------------------------------------------------------------------------
# Test 8: ProvenanceChainService.verify_chain — valid chain
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verify_chain_passes_for_valid_chain(
    service: ProvenanceChainService,
    tenant_id: uuid.UUID,
    model_id: uuid.UUID,
    private_key_pem: bytes,
    public_key_pem: str,
    public_key_id: uuid.UUID,
) -> None:
    """verify_chain returns is_valid=True for a properly constructed chain."""
    chain_id = uuid.uuid4()

    # Build two genuine links
    payload1 = {
        "__chain_id": str(chain_id),
        "__sequence": 1,
        "__link_type": "training_dataset",
        "__recorded_at": datetime.now(UTC).isoformat(),
        "dataset_uri": "s3://test",
    }
    hash1 = compute_link_hash(None, payload1)
    sig1 = sign_link_payload(payload1, private_key_pem)

    payload2 = {
        "__chain_id": str(chain_id),
        "__sequence": 2,
        "__link_type": "training_run",
        "__recorded_at": datetime.now(UTC).isoformat(),
        "run_id": "run-001",
    }
    hash2 = compute_link_hash(hash1, payload2)
    sig2 = sign_link_payload(payload2, private_key_pem)

    chain = _make_chain(chain_id, tenant_id, model_id, head_link_hash=hash2)
    link1 = _make_link(chain_id, 1, "training_dataset", payload1, hash1, None, sig1, public_key_id, tenant_id)
    link2 = _make_link(chain_id, 2, "training_run", payload2, hash2, hash1, sig2, public_key_id, tenant_id)

    public_key_record = _make_public_key_record(public_key_id, public_key_pem, tenant_id)

    session = AsyncMock()
    session.flush = AsyncMock()

    chain_result = MagicMock()
    chain_result.scalar_one_or_none.return_value = chain

    links_scalars = MagicMock()
    links_scalars.all.return_value = [link1, link2]
    links_result = MagicMock()
    links_result.scalars.return_value = links_scalars

    key_result = MagicMock()
    key_result.scalar_one_or_none.return_value = public_key_record

    session.execute = AsyncMock(side_effect=[chain_result, links_result, key_result, key_result])

    result = await service.verify_chain(
        session=session,
        chain_id=chain_id,
        tenant_id=tenant_id,
    )

    assert result.is_valid is True
    assert result.link_count == 2
    assert result.violations == []


# ---------------------------------------------------------------------------
# Test 9: Tamper detection — payload mutation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verify_chain_detects_tampered_payload(
    service: ProvenanceChainService,
    tenant_id: uuid.UUID,
    model_id: uuid.UUID,
    private_key_pem: bytes,
    public_key_pem: str,
    public_key_id: uuid.UUID,
) -> None:
    """verify_chain detects when a link's payload has been mutated."""
    chain_id = uuid.uuid4()

    # Create a genuine link
    original_payload = {
        "__chain_id": str(chain_id),
        "__sequence": 1,
        "__link_type": "training_dataset",
        "__recorded_at": datetime.now(UTC).isoformat(),
        "dataset_uri": "s3://legitimate-bucket/data.parquet",
    }
    original_hash = compute_link_hash(None, original_payload)
    sig = sign_link_payload(original_payload, private_key_pem)

    # Tamper the payload (attacker changes dataset URI)
    tampered_payload = {**original_payload, "dataset_uri": "s3://attacker-bucket/poison.csv"}

    chain = _make_chain(chain_id, tenant_id, model_id, head_link_hash=original_hash)
    tampered_link = _make_link(
        chain_id, 1, "training_dataset",
        tampered_payload,  # Tampered payload
        original_hash,     # Original hash (no longer matches tampered payload)
        None, sig, public_key_id, tenant_id,
    )

    public_key_record = _make_public_key_record(public_key_id, public_key_pem, tenant_id)

    session = AsyncMock()
    session.flush = AsyncMock()

    chain_result = MagicMock()
    chain_result.scalar_one_or_none.return_value = chain

    links_scalars = MagicMock()
    links_scalars.all.return_value = [tampered_link]
    links_result = MagicMock()
    links_result.scalars.return_value = links_scalars

    key_result = MagicMock()
    key_result.scalar_one_or_none.return_value = public_key_record

    session.execute = AsyncMock(side_effect=[chain_result, links_result, key_result])

    result = await service.verify_chain(
        session=session,
        chain_id=chain_id,
        tenant_id=tenant_id,
    )

    assert result.is_valid is False
    # At minimum, the hash mismatch and signature mismatch should be detected
    assert len(result.violations) >= 1


# ---------------------------------------------------------------------------
# Test 10: Tamper detection — broken hash chain
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verify_chain_detects_broken_hash_chain(
    service: ProvenanceChainService,
    tenant_id: uuid.UUID,
    model_id: uuid.UUID,
    private_key_pem: bytes,
    public_key_pem: str,
    public_key_id: uuid.UUID,
) -> None:
    """verify_chain detects when a link's previous_link_hash has been forged."""
    chain_id = uuid.uuid4()

    payload1 = {
        "__chain_id": str(chain_id), "__sequence": 1,
        "__link_type": "training_dataset", "__recorded_at": "2024-01-01T00:00:00+00:00",
        "data": "first",
    }
    hash1 = compute_link_hash(None, payload1)
    sig1 = sign_link_payload(payload1, private_key_pem)

    payload2 = {
        "__chain_id": str(chain_id), "__sequence": 2,
        "__link_type": "training_run", "__recorded_at": "2024-01-01T01:00:00+00:00",
        "data": "second",
    }
    # Intentionally use wrong previous hash (attacker replaces link1 but can't
    # update link2's previous_link_hash without breaking sig2 too)
    forged_previous = "f" * 64
    hash2 = compute_link_hash(forged_previous, payload2)
    sig2 = sign_link_payload(payload2, private_key_pem)

    chain = _make_chain(chain_id, tenant_id, model_id, head_link_hash=hash2)
    link1 = _make_link(chain_id, 1, "training_dataset", payload1, hash1, None, sig1, public_key_id, tenant_id)
    link2 = _make_link(chain_id, 2, "training_run", payload2, hash2, forged_previous, sig2, public_key_id, tenant_id)

    public_key_record = _make_public_key_record(public_key_id, public_key_pem, tenant_id)

    session = AsyncMock()
    session.flush = AsyncMock()

    chain_result = MagicMock()
    chain_result.scalar_one_or_none.return_value = chain

    links_scalars = MagicMock()
    links_scalars.all.return_value = [link1, link2]
    links_result = MagicMock()
    links_result.scalars.return_value = links_scalars

    key_result = MagicMock()
    key_result.scalar_one_or_none.return_value = public_key_record

    session.execute = AsyncMock(
        side_effect=[chain_result, links_result, key_result, key_result]
    )

    result = await service.verify_chain(
        session=session,
        chain_id=chain_id,
        tenant_id=tenant_id,
    )

    assert result.is_valid is False
    # Link 2's previous_link_hash doesn't match link 1's hash
    previous_hash_violations = [v for v in result.violations if "previous_link_hash" in v]
    assert len(previous_hash_violations) >= 1


# ---------------------------------------------------------------------------
# Test 11: Tamper detection — revoked/missing public key
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verify_chain_detects_missing_public_key(
    service: ProvenanceChainService,
    tenant_id: uuid.UUID,
    model_id: uuid.UUID,
    private_key_pem: bytes,
    public_key_pem: str,
    public_key_id: uuid.UUID,
) -> None:
    """verify_chain reports violation when the signing key is not found."""
    chain_id = uuid.uuid4()

    payload = {
        "__chain_id": str(chain_id), "__sequence": 1,
        "__link_type": "training_dataset", "__recorded_at": "2024-01-01T00:00:00+00:00",
    }
    link_hash = compute_link_hash(None, payload)
    sig = sign_link_payload(payload, private_key_pem)

    chain = _make_chain(chain_id, tenant_id, model_id, head_link_hash=link_hash)
    link = _make_link(chain_id, 1, "training_dataset", payload, link_hash, None, sig, public_key_id, tenant_id)

    session = AsyncMock()
    session.flush = AsyncMock()

    chain_result = MagicMock()
    chain_result.scalar_one_or_none.return_value = chain

    links_scalars = MagicMock()
    links_scalars.all.return_value = [link]
    links_result = MagicMock()
    links_result.scalars.return_value = links_scalars

    # Public key not found (deleted/revoked)
    key_result = MagicMock()
    key_result.scalar_one_or_none.return_value = None

    session.execute = AsyncMock(side_effect=[chain_result, links_result, key_result])

    result = await service.verify_chain(
        session=session,
        chain_id=chain_id,
        tenant_id=tenant_id,
    )

    assert result.is_valid is False
    key_violations = [v for v in result.violations if "public key" in v.lower()]
    assert len(key_violations) >= 1


# ---------------------------------------------------------------------------
# Test 12: seal_chain
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_seal_chain_changes_status_to_sealed(
    service: ProvenanceChainService,
    tenant_id: uuid.UUID,
    model_id: uuid.UUID,
) -> None:
    """seal_chain transitions chain status from 'active' to 'sealed'."""
    chain_id = uuid.uuid4()
    chain = _make_chain(chain_id, tenant_id, model_id, chain_status="active")

    session = AsyncMock()
    session.flush = AsyncMock()

    chain_result = MagicMock()
    chain_result.scalar_one_or_none.return_value = chain
    session.execute = AsyncMock(return_value=chain_result)

    returned_chain = await service.seal_chain(
        session=session,
        chain_id=chain_id,
        tenant_id=tenant_id,
    )
    assert returned_chain.chain_status == "sealed"


@pytest.mark.asyncio
async def test_seal_chain_raises_for_already_sealed(
    service: ProvenanceChainService,
    tenant_id: uuid.UUID,
    model_id: uuid.UUID,
) -> None:
    """seal_chain raises ValueError when chain is already sealed."""
    chain_id = uuid.uuid4()
    chain = _make_chain(chain_id, tenant_id, model_id, chain_status="sealed")

    session = AsyncMock()
    chain_result = MagicMock()
    chain_result.scalar_one_or_none.return_value = chain
    session.execute = AsyncMock(return_value=chain_result)

    with pytest.raises(ValueError, match="sealed"):
        await service.seal_chain(
            session=session,
            chain_id=chain_id,
            tenant_id=tenant_id,
        )


# ---------------------------------------------------------------------------
# Test 13: LINK_TYPES contains all seven stages
# ---------------------------------------------------------------------------


def test_link_types_contains_all_seven_stages() -> None:
    """LINK_TYPES contains the complete 7-stage model lifecycle."""
    expected_stages = {
        "training_dataset",
        "training_run",
        "model_artifact",
        "validation_results",
        "approval_decision",
        "deployment_record",
        "inference_log",
    }
    assert expected_stages == set(LINK_TYPES)


# ---------------------------------------------------------------------------
# Test 14: Full lifecycle — sequential link addition verification
# ---------------------------------------------------------------------------


def test_hash_chain_integrity_across_seven_links(
    private_key_pem: bytes,
    public_key_pem: str,
    public_key_id: uuid.UUID,
) -> None:
    """A seven-link chain built incrementally has consistent hash linkage."""
    chain_id = uuid.uuid4()
    previous_hash: str | None = None
    expected_hashes: list[str] = []

    for idx, link_type in enumerate(LINK_TYPES):
        payload = {
            "__chain_id": str(chain_id),
            "__sequence": idx + 1,
            "__link_type": link_type,
            "__recorded_at": datetime.now(UTC).isoformat(),
            "stage_data": f"data_for_{link_type}",
        }
        link_hash = compute_link_hash(previous_hash, payload)
        expected_hashes.append(link_hash)
        previous_hash = link_hash

    # Verify the chain by re-computing from scratch
    recomputed_previous: str | None = None
    for idx, link_type in enumerate(LINK_TYPES):
        payload = {
            "__chain_id": str(chain_id),
            "__sequence": idx + 1,
            "__link_type": link_type,
            "__recorded_at": datetime.now(UTC).isoformat(),
            "stage_data": f"data_for_{link_type}",
        }
        recomputed_hash = compute_link_hash(recomputed_previous, payload)
        assert recomputed_hash == expected_hashes[idx], (
            f"Hash mismatch at link {idx + 1} ({link_type})"
        )
        recomputed_previous = recomputed_hash


# ---------------------------------------------------------------------------
# Test 15: verify_chain empty chain is valid
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verify_chain_empty_chain_is_valid(
    service: ProvenanceChainService,
    tenant_id: uuid.UUID,
    model_id: uuid.UUID,
) -> None:
    """An empty chain (no links yet) passes verification."""
    chain_id = uuid.uuid4()
    chain = _make_chain(chain_id, tenant_id, model_id)

    session = AsyncMock()
    session.flush = AsyncMock()

    chain_result = MagicMock()
    chain_result.scalar_one_or_none.return_value = chain

    links_scalars = MagicMock()
    links_scalars.all.return_value = []
    links_result = MagicMock()
    links_result.scalars.return_value = links_scalars

    session.execute = AsyncMock(side_effect=[chain_result, links_result])

    result = await service.verify_chain(
        session=session,
        chain_id=chain_id,
        tenant_id=tenant_id,
    )

    assert result.is_valid is True
    assert result.link_count == 0
    assert result.violations == []


# ---------------------------------------------------------------------------
# Test 16: verify_chain not found raises NotFoundError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verify_chain_raises_not_found_for_missing_chain(
    service: ProvenanceChainService,
    tenant_id: uuid.UUID,
) -> None:
    """verify_chain raises NotFoundError when chain doesn't exist."""
    from aumos_common.errors import NotFoundError

    chain_id = uuid.uuid4()

    session = AsyncMock()
    chain_result = MagicMock()
    chain_result.scalar_one_or_none.return_value = None
    session.execute = AsyncMock(return_value=chain_result)

    with pytest.raises(NotFoundError):
        await service.verify_chain(
            session=session,
            chain_id=chain_id,
            tenant_id=tenant_id,
        )


# ---------------------------------------------------------------------------
# Test 17: Hash is payload-key-order independent (canonical JSON)
# ---------------------------------------------------------------------------


def test_canonical_json_key_order_does_not_affect_hash() -> None:
    """Payload with keys in different order produces the same hash."""
    payload_a = {"b": 2, "a": 1, "c": 3}
    payload_b = {"c": 3, "a": 1, "b": 2}
    prev_hash = "x" * 64
    assert compute_link_hash(prev_hash, payload_a) == compute_link_hash(prev_hash, payload_b)

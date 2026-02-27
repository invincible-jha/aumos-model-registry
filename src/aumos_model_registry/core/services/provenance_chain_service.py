"""Provenance Chain Service — cryptographic chain of custody for AI models.

Provides Ed25519-signed, SHA-256 hash-chained provenance records for AI model
lifecycle stages. Each link in the chain covers one lifecycle stage:

  TrainingDataset → TrainingRun → ModelArtifact → ValidationResults
  → ApprovalDecision → DeploymentRecord → InferenceLog

Chain integrity is verified by re-computing the hash chain and verifying all
Ed25519 signatures against the registered public key.

Architecture:
- All signing uses Ed25519 from the ``cryptography`` package (no custom crypto)
- Hash chaining: SHA-256(previous_link_hash_bytes + canonical_json_bytes)
- Canonical JSON: json.dumps with sorted keys and no whitespace
- Signatures: Ed25519 sign over canonical_json bytes (NOT the hash)
- Public keys are stored in reg_provenance_public_keys (PEM format)
- Private keys are NEVER persisted — callers pass them in for signing

This service is intentionally stateless (no DB session in constructor).
Callers inject the DB session via each method call, following the
hexagonal architecture pattern used by other AumOS services.
"""

import base64
import hashlib
import json
import uuid
from datetime import UTC, datetime
from typing import Any

from aumos_common.errors import NotFoundError
from aumos_common.observability import get_logger

from aumos_model_registry.core.models import (
    ModelProvenanceChain,
    ProvenanceChainLink,
    ProvenancePublicKey,
)

logger = get_logger(__name__)

# Valid link types in chain order
LINK_TYPES = (
    "training_dataset",
    "training_run",
    "model_artifact",
    "validation_results",
    "approval_decision",
    "deployment_record",
    "inference_log",
)

# Genesis link marker
_GENESIS_HASH = "0" * 64


def _canonical_json(payload: dict[str, Any]) -> bytes:
    """Serialize payload to canonical JSON bytes (sorted keys, no whitespace).

    Args:
        payload: Dictionary to serialize.

    Returns:
        UTF-8 encoded canonical JSON bytes.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def compute_link_hash(
    previous_link_hash: str | None,
    payload: dict[str, Any],
) -> str:
    """Compute the SHA-256 hash for a chain link.

    Hash = SHA-256(previous_link_hash_utf8 || canonical_payload_bytes)
    Where || denotes byte concatenation and previous_link_hash is the
    ASCII string encoding of the hex hash (or 64 zeros for genesis).

    Args:
        previous_link_hash: Hex hash of the predecessor link, or None for genesis.
        payload: Link payload dictionary.

    Returns:
        Lower-case hex SHA-256 hash string (64 characters).
    """
    prev_hash_bytes = (previous_link_hash or _GENESIS_HASH).encode("ascii")
    payload_bytes = _canonical_json(payload)
    digest = hashlib.sha256(prev_hash_bytes + payload_bytes).hexdigest()
    return digest


def sign_link_payload(
    payload: dict[str, Any],
    private_key_pem: bytes,
) -> str:
    """Sign the canonical JSON of a link payload using Ed25519.

    Args:
        payload: The link payload to sign.
        private_key_pem: PEM-encoded Ed25519 private key bytes.

    Returns:
        Base64-encoded Ed25519 signature string.

    Raises:
        ValueError: If the private key is not a valid Ed25519 key.
    """
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives.serialization import load_pem_private_key
    except ImportError as exc:
        raise ImportError(
            "cryptography package is required for provenance chain signing. "
            "Install with: pip install cryptography>=42.0.0"
        ) from exc

    private_key = load_pem_private_key(private_key_pem, password=None)
    if not isinstance(private_key, Ed25519PrivateKey):
        raise ValueError("Private key must be an Ed25519 key")

    payload_bytes = _canonical_json(payload)
    raw_signature = private_key.sign(payload_bytes)
    return base64.b64encode(raw_signature).decode("ascii")


def verify_link_signature(
    payload: dict[str, Any],
    signature_b64: str,
    public_key_pem: str,
) -> bool:
    """Verify an Ed25519 signature against a link payload.

    Args:
        payload: The link payload that was signed.
        signature_b64: Base64-encoded Ed25519 signature.
        public_key_pem: PEM-encoded Ed25519 public key string.

    Returns:
        True if the signature is valid, False otherwise.
    """
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        from cryptography.hazmat.primitives.serialization import load_pem_public_key
    except ImportError as exc:
        raise ImportError(
            "cryptography package is required for provenance chain verification."
        ) from exc

    public_key = load_pem_public_key(public_key_pem.encode("utf-8"))
    if not isinstance(public_key, Ed25519PublicKey):
        return False

    payload_bytes = _canonical_json(payload)
    raw_signature = base64.b64decode(signature_b64)

    try:
        public_key.verify(raw_signature, payload_bytes)
        return True
    except InvalidSignature:
        return False


def generate_ed25519_keypair() -> tuple[bytes, str]:
    """Generate a new Ed25519 keypair for signing provenance chain links.

    This is a utility for testing and initial key provisioning. In production,
    keys should be generated and stored in a KMS, not this function.

    Returns:
        Tuple of (private_key_pem_bytes, public_key_pem_str).
    """
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        NoEncryption,
        PublicFormat,
        PrivateFormat,
    )

    private_key = Ed25519PrivateKey.generate()
    private_pem = private_key.private_bytes(
        encoding=Encoding.PEM,
        format=PrivateFormat.PKCS8,
        encryption_algorithm=NoEncryption(),
    )
    public_pem = private_key.public_key().public_bytes(
        encoding=Encoding.PEM,
        format=PublicFormat.SubjectPublicKeyInfo,
    ).decode("utf-8")
    return private_pem, public_pem


class ProvenanceChainResult:
    """Result of a chain verification operation.

    Attributes:
        chain_id: UUID of the chain that was verified.
        is_valid: Whether the chain passed all integrity checks.
        link_count: Number of links verified.
        violations: List of violation messages (empty if valid).
        verified_at: When verification was performed.
    """

    def __init__(
        self,
        chain_id: uuid.UUID,
        is_valid: bool,
        link_count: int,
        violations: list[str],
        verified_at: datetime,
    ) -> None:
        """Initialize ProvenanceChainResult.

        Args:
            chain_id: UUID of the verified chain.
            is_valid: Whether the chain is cryptographically intact.
            link_count: Number of links in the chain.
            violations: Descriptions of any violations found.
            verified_at: Timestamp of verification.
        """
        self.chain_id = chain_id
        self.is_valid = is_valid
        self.link_count = link_count
        self.violations = violations
        self.verified_at = verified_at


class ProvenanceChainService:
    """Service for creating and verifying cryptographic model provenance chains.

    All methods are async-first. The service does not hold a DB session —
    the caller passes a session to each method, following the AumOS convention.

    Key operations:
    - create_chain: Start a new provenance chain for a model
    - add_link: Append a signed, hash-linked entry to an existing chain
    - verify_chain: Re-compute hashes and verify all signatures
    - get_chain: Retrieve a chain with all its links
    - register_public_key: Register an Ed25519 public key for a tenant
    """

    async def create_chain(
        self,
        session: Any,
        tenant_id: uuid.UUID,
        model_id: uuid.UUID,
        model_version_id: uuid.UUID | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ModelProvenanceChain:
        """Create a new provenance chain for a model.

        Args:
            session: Async SQLAlchemy session.
            tenant_id: Owning tenant UUID.
            model_id: UUID of the model this chain tracks.
            model_version_id: Optional specific model version UUID.
            metadata: Optional additional context metadata.

        Returns:
            The newly created ModelProvenanceChain instance.
        """
        chain = ModelProvenanceChain(
            tenant_id=tenant_id,
            model_id=model_id,
            model_version_id=model_version_id,
            chain_status="active",
            is_verified=False,
            head_link_hash=None,
            metadata_=metadata or {},
        )
        session.add(chain)
        await session.flush()

        logger.info(
            "Provenance chain created",
            chain_id=str(chain.id),
            tenant_id=str(tenant_id),
            model_id=str(model_id),
        )
        return chain

    async def add_link(
        self,
        session: Any,
        chain_id: uuid.UUID,
        tenant_id: uuid.UUID,
        link_type: str,
        payload: dict[str, Any],
        private_key_pem: bytes,
        public_key_id: uuid.UUID,
        actor_id: uuid.UUID | None = None,
    ) -> ProvenanceChainLink:
        """Append a cryptographically signed link to an existing chain.

        Computes the SHA-256 hash linking this entry to the previous one,
        signs the payload with Ed25519, and persists the link.

        Args:
            session: Async SQLAlchemy session.
            chain_id: UUID of the chain to extend.
            tenant_id: Owning tenant UUID (used to validate chain ownership).
            link_type: One of the LINK_TYPES values.
            payload: Structured data for this lifecycle stage.
            private_key_pem: Ed25519 private key bytes for signing.
            public_key_id: UUID of the registered ProvenancePublicKey.
            actor_id: Optional UUID of the user/service creating this link.

        Returns:
            The newly created ProvenanceChainLink.

        Raises:
            NotFoundError: If the chain does not exist.
            ValueError: If the link_type is invalid or the chain is sealed.
        """
        from sqlalchemy import select

        # Load chain
        stmt = select(ModelProvenanceChain).where(
            ModelProvenanceChain.id == chain_id,
            ModelProvenanceChain.tenant_id == tenant_id,
        )
        result = await session.execute(stmt)
        chain: ModelProvenanceChain | None = result.scalar_one_or_none()
        if chain is None:
            raise NotFoundError(f"Provenance chain {chain_id} not found")

        if chain.chain_status != "active":
            raise ValueError(
                f"Cannot append to chain {chain_id}: status is '{chain.chain_status}'"
            )

        if link_type not in LINK_TYPES:
            raise ValueError(
                f"Invalid link_type '{link_type}'. Must be one of: {LINK_TYPES}"
            )

        # Determine sequence number
        from sqlalchemy import func

        count_stmt = select(func.count()).where(
            ProvenanceChainLink.chain_id == chain_id
        )
        count_result = await session.execute(count_stmt)
        existing_count: int = count_result.scalar_one()
        sequence_number = existing_count + 1

        # Enrich payload with standard provenance metadata
        enriched_payload = {
            **payload,
            "__chain_id": str(chain_id),
            "__sequence": sequence_number,
            "__link_type": link_type,
            "__recorded_at": datetime.now(UTC).isoformat(),
        }

        # Compute hash chain
        previous_hash = chain.head_link_hash
        link_hash = compute_link_hash(previous_hash, enriched_payload)

        # Sign the payload
        signature = sign_link_payload(enriched_payload, private_key_pem)

        link = ProvenanceChainLink(
            tenant_id=tenant_id,
            chain_id=chain_id,
            sequence_number=sequence_number,
            link_type=link_type,
            payload=enriched_payload,
            link_hash=link_hash,
            previous_link_hash=previous_hash,
            signature=signature,
            signed_by=public_key_id,
            actor_id=actor_id,
        )
        session.add(link)

        # Update chain head
        chain.head_link_hash = link_hash

        await session.flush()

        logger.info(
            "Provenance chain link appended",
            chain_id=str(chain_id),
            sequence_number=sequence_number,
            link_type=link_type,
            link_hash=link_hash,
        )
        return link

    async def verify_chain(
        self,
        session: Any,
        chain_id: uuid.UUID,
        tenant_id: uuid.UUID,
    ) -> ProvenanceChainResult:
        """Verify the cryptographic integrity of a provenance chain.

        Checks:
        1. Hash continuity: each link's previous_link_hash matches the prior link's hash
        2. Hash correctness: recomputes each link's hash and compares to stored value
        3. Signature validity: verifies Ed25519 signature on each link payload
        4. Sequence integrity: sequence numbers are contiguous starting from 1

        Args:
            session: Async SQLAlchemy session.
            chain_id: UUID of the chain to verify.
            tenant_id: Owning tenant UUID.

        Returns:
            ProvenanceChainResult with is_valid=True if all checks pass.

        Raises:
            NotFoundError: If the chain does not exist.
        """
        from sqlalchemy import select

        # Load chain
        stmt = select(ModelProvenanceChain).where(
            ModelProvenanceChain.id == chain_id,
            ModelProvenanceChain.tenant_id == tenant_id,
        )
        result = await session.execute(stmt)
        chain: ModelProvenanceChain | None = result.scalar_one_or_none()
        if chain is None:
            raise NotFoundError(f"Provenance chain {chain_id} not found")

        # Load all links in order
        links_stmt = (
            select(ProvenanceChainLink)
            .where(ProvenanceChainLink.chain_id == chain_id)
            .order_by(ProvenanceChainLink.sequence_number)
        )
        links_result = await session.execute(links_stmt)
        links: list[ProvenanceChainLink] = list(links_result.scalars().all())

        violations: list[str] = []
        now = datetime.now(UTC)

        if not links:
            return ProvenanceChainResult(
                chain_id=chain_id,
                is_valid=True,
                link_count=0,
                violations=[],
                verified_at=now,
            )

        # Cache public keys to avoid repeated DB lookups
        public_key_cache: dict[uuid.UUID, str | None] = {}

        async def _get_public_key_pem(key_id: uuid.UUID) -> str | None:
            if key_id in public_key_cache:
                return public_key_cache[key_id]
            key_stmt = select(ProvenancePublicKey).where(
                ProvenancePublicKey.id == key_id,
                ProvenancePublicKey.tenant_id == tenant_id,
                ProvenancePublicKey.is_active.is_(True),
            )
            key_result = await session.execute(key_stmt)
            key_record: ProvenancePublicKey | None = key_result.scalar_one_or_none()
            pem = key_record.public_key_pem if key_record else None
            public_key_cache[key_id] = pem
            return pem

        previous_hash: str | None = None

        for idx, link in enumerate(links):
            expected_sequence = idx + 1

            # Check 1: Sequence integrity
            if link.sequence_number != expected_sequence:
                violations.append(
                    f"Link at position {idx} has sequence_number {link.sequence_number}, "
                    f"expected {expected_sequence}"
                )

            # Check 2: Previous hash continuity
            if link.previous_link_hash != previous_hash:
                violations.append(
                    f"Link {link.sequence_number} previous_link_hash mismatch: "
                    f"stored='{link.previous_link_hash}', expected='{previous_hash}'"
                )

            # Check 3: Hash recomputation
            recomputed_hash = compute_link_hash(previous_hash, link.payload)
            if link.link_hash != recomputed_hash:
                violations.append(
                    f"Link {link.sequence_number} hash mismatch: "
                    f"stored='{link.link_hash}', recomputed='{recomputed_hash}'"
                )

            # Check 4: Ed25519 signature
            public_key_pem = await _get_public_key_pem(link.signed_by)
            if public_key_pem is None:
                violations.append(
                    f"Link {link.sequence_number}: public key {link.signed_by} not found "
                    "or is revoked"
                )
            else:
                signature_valid = verify_link_signature(
                    link.payload,
                    link.signature,
                    public_key_pem,
                )
                if not signature_valid:
                    violations.append(
                        f"Link {link.sequence_number}: Ed25519 signature is invalid"
                    )

            previous_hash = link.link_hash

        is_valid = len(violations) == 0

        # Update chain verification status
        chain.is_verified = is_valid

        logger.info(
            "Provenance chain verification complete",
            chain_id=str(chain_id),
            is_valid=is_valid,
            link_count=len(links),
            violation_count=len(violations),
        )

        return ProvenanceChainResult(
            chain_id=chain_id,
            is_valid=is_valid,
            link_count=len(links),
            violations=violations,
            verified_at=now,
        )

    async def get_chain(
        self,
        session: Any,
        chain_id: uuid.UUID,
        tenant_id: uuid.UUID,
    ) -> ModelProvenanceChain | None:
        """Retrieve a provenance chain with all its links.

        Args:
            session: Async SQLAlchemy session.
            chain_id: UUID of the chain to retrieve.
            tenant_id: Owning tenant UUID.

        Returns:
            ModelProvenanceChain with links loaded, or None if not found.
        """
        from sqlalchemy import select
        from sqlalchemy.orm import selectinload

        stmt = (
            select(ModelProvenanceChain)
            .where(
                ModelProvenanceChain.id == chain_id,
                ModelProvenanceChain.tenant_id == tenant_id,
            )
            .options(selectinload(ModelProvenanceChain.links))
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    async def register_public_key(
        self,
        session: Any,
        tenant_id: uuid.UUID,
        key_label: str,
        public_key_pem: str,
    ) -> ProvenancePublicKey:
        """Register an Ed25519 public key for signing provenance chain links.

        Args:
            session: Async SQLAlchemy session.
            tenant_id: Owning tenant UUID.
            key_label: Human-readable label for the key.
            public_key_pem: PEM-encoded Ed25519 public key string.

        Returns:
            The newly created ProvenancePublicKey record.

        Raises:
            ValueError: If the provided key is not a valid Ed25519 public key.
        """
        # Validate the public key before persisting
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
            from cryptography.hazmat.primitives.serialization import load_pem_public_key

            key = load_pem_public_key(public_key_pem.encode("utf-8"))
            if not isinstance(key, Ed25519PublicKey):
                raise ValueError("Key must be an Ed25519 public key")
        except Exception as exc:
            if "Ed25519" in str(exc) or "Key must be" in str(exc):
                raise ValueError(f"Invalid Ed25519 public key: {exc}") from exc
            raise ValueError(f"Failed to parse public key PEM: {exc}") from exc

        key_record = ProvenancePublicKey(
            tenant_id=tenant_id,
            key_label=key_label,
            public_key_pem=public_key_pem,
            is_active=True,
            revoked_at=None,
        )
        session.add(key_record)
        await session.flush()

        logger.info(
            "Provenance public key registered",
            key_id=str(key_record.id),
            tenant_id=str(tenant_id),
            key_label=key_label,
        )
        return key_record

    async def seal_chain(
        self,
        session: Any,
        chain_id: uuid.UUID,
        tenant_id: uuid.UUID,
    ) -> ModelProvenanceChain:
        """Seal a provenance chain, preventing further link additions.

        A sealed chain is read-only — no new links can be appended.
        Typically called after all lifecycle stages are complete.

        Args:
            session: Async SQLAlchemy session.
            chain_id: UUID of the chain to seal.
            tenant_id: Owning tenant UUID.

        Returns:
            The updated ModelProvenanceChain.

        Raises:
            NotFoundError: If the chain does not exist.
            ValueError: If the chain is already sealed or revoked.
        """
        from sqlalchemy import select

        stmt = select(ModelProvenanceChain).where(
            ModelProvenanceChain.id == chain_id,
            ModelProvenanceChain.tenant_id == tenant_id,
        )
        result = await session.execute(stmt)
        chain: ModelProvenanceChain | None = result.scalar_one_or_none()
        if chain is None:
            raise NotFoundError(f"Provenance chain {chain_id} not found")

        if chain.chain_status != "active":
            raise ValueError(
                f"Chain {chain_id} is already '{chain.chain_status}', cannot seal"
            )

        chain.chain_status = "sealed"
        await session.flush()

        logger.info("Provenance chain sealed", chain_id=str(chain_id))
        return chain

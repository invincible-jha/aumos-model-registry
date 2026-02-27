"""FastAPI routes for the Model Provenance Chain API (P1.2).

Endpoints:
- POST   /provenance/chains              — Create a new provenance chain
- POST   /provenance/chains/{chain_id}/links — Append a signed link
- GET    /provenance/chains/{chain_id}   — Retrieve chain with all links
- POST   /provenance/chains/{chain_id}/verify — Verify chain integrity
- POST   /provenance/keys               — Register an Ed25519 public key
"""

import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from aumos_common.auth import TenantContext, get_current_tenant
from aumos_common.database import get_db_session
from aumos_common.errors import NotFoundError
from aumos_common.observability import get_logger

from aumos_model_registry.core.services.provenance_chain_service import (
    LINK_TYPES,
    ProvenanceChainService,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/provenance", tags=["provenance"])


# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------


class CreateChainRequest(BaseModel):
    """Request body for creating a new provenance chain."""

    model_id: uuid.UUID = Field(..., description="UUID of the model to track")
    model_version_id: uuid.UUID | None = Field(
        None, description="Specific model version UUID, if applicable"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context metadata for the chain",
    )


class ChainLinkResponse(BaseModel):
    """Serialized provenance chain link."""

    id: uuid.UUID
    chain_id: uuid.UUID
    sequence_number: int
    link_type: str
    payload: dict[str, Any]
    link_hash: str
    previous_link_hash: str | None
    signature: str
    signed_by: uuid.UUID
    actor_id: uuid.UUID | None

    model_config = {"from_attributes": True}


class ChainResponse(BaseModel):
    """Serialized provenance chain."""

    id: uuid.UUID
    tenant_id: uuid.UUID
    model_id: uuid.UUID
    model_version_id: uuid.UUID | None
    chain_status: str
    is_verified: bool
    head_link_hash: str | None
    metadata: dict[str, Any] = Field(alias="metadata_")
    links: list[ChainLinkResponse] = Field(default_factory=list)

    model_config = {"from_attributes": True, "populate_by_name": True}


class AddLinkRequest(BaseModel):
    """Request body for appending a link to a provenance chain."""

    link_type: str = Field(
        ...,
        description=(
            f"Lifecycle stage type. Must be one of: {LINK_TYPES}"
        ),
    )
    payload: dict[str, Any] = Field(
        ...,
        description="Structured data describing this lifecycle stage",
    )
    private_key_pem: str = Field(
        ...,
        description="PEM-encoded Ed25519 private key for signing (NOT persisted)",
    )
    public_key_id: uuid.UUID = Field(
        ...,
        description="UUID of the registered ProvenancePublicKey to reference",
    )
    actor_id: uuid.UUID | None = Field(
        None,
        description="UUID of the user or service creating this link",
    )


class AddLinkResponse(BaseModel):
    """Response after appending a chain link."""

    id: uuid.UUID
    chain_id: uuid.UUID
    sequence_number: int
    link_type: str
    link_hash: str
    previous_link_hash: str | None
    signed_by: uuid.UUID

    model_config = {"from_attributes": True}


class VerifyChainResponse(BaseModel):
    """Result of a chain verification request."""

    chain_id: uuid.UUID
    is_valid: bool
    link_count: int
    violations: list[str]
    verified_at: str


class RegisterPublicKeyRequest(BaseModel):
    """Request body for registering an Ed25519 public key."""

    key_label: str = Field(..., max_length=255, description="Human-readable key label")
    public_key_pem: str = Field(..., description="PEM-encoded Ed25519 public key")


class PublicKeyResponse(BaseModel):
    """Response after registering a public key."""

    id: uuid.UUID
    tenant_id: uuid.UUID
    key_label: str
    is_active: bool

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Dependency injection
# ---------------------------------------------------------------------------


def get_provenance_service() -> ProvenanceChainService:
    """Provide a ProvenanceChainService instance.

    Returns:
        Singleton-like ProvenanceChainService (stateless, safe to create per request).
    """
    return ProvenanceChainService()


ProvServiceDep = Annotated[ProvenanceChainService, Depends(get_provenance_service)]
TenantDep = Annotated[TenantContext, Depends(get_current_tenant)]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/chains", status_code=status.HTTP_201_CREATED, response_model=ChainResponse)
async def create_chain(
    request: CreateChainRequest,
    tenant: TenantDep,
    service: ProvServiceDep,
    session: Annotated[Any, Depends(get_db_session)],
) -> Any:
    """Create a new provenance chain for an AI model.

    The chain begins with no links. Append links via POST /chains/{id}/links.
    """
    chain = await service.create_chain(
        session=session,
        tenant_id=tenant.tenant_id,
        model_id=request.model_id,
        model_version_id=request.model_version_id,
        metadata=request.metadata,
    )
    return chain


@router.post(
    "/chains/{chain_id}/links",
    status_code=status.HTTP_201_CREATED,
    response_model=AddLinkResponse,
)
async def add_link(
    chain_id: uuid.UUID,
    request: AddLinkRequest,
    tenant: TenantDep,
    service: ProvServiceDep,
    session: Annotated[Any, Depends(get_db_session)],
) -> Any:
    """Append a signed, hash-linked entry to an existing provenance chain.

    The private key is used for signing only and is never persisted.
    Ensure public_key_id references a registered key for the same tenant.
    """
    try:
        link = await service.add_link(
            session=session,
            chain_id=chain_id,
            tenant_id=tenant.tenant_id,
            link_type=request.link_type,
            payload=request.payload,
            private_key_pem=request.private_key_pem.encode("utf-8"),
            public_key_id=request.public_key_id,
            actor_id=request.actor_id,
        )
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    return link


@router.get("/chains/{chain_id}", response_model=ChainResponse)
async def get_chain(
    chain_id: uuid.UUID,
    tenant: TenantDep,
    service: ProvServiceDep,
    session: Annotated[Any, Depends(get_db_session)],
) -> Any:
    """Retrieve a provenance chain with all its links."""
    chain = await service.get_chain(
        session=session,
        chain_id=chain_id,
        tenant_id=tenant.tenant_id,
    )
    if chain is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Provenance chain {chain_id} not found",
        )
    return chain


@router.post("/chains/{chain_id}/verify", response_model=VerifyChainResponse)
async def verify_chain(
    chain_id: uuid.UUID,
    tenant: TenantDep,
    service: ProvServiceDep,
    session: Annotated[Any, Depends(get_db_session)],
) -> Any:
    """Verify the cryptographic integrity of a provenance chain.

    Checks hash continuity, hash recomputation, Ed25519 signatures,
    and sequence number integrity.
    """
    try:
        result = await service.verify_chain(
            session=session,
            chain_id=chain_id,
            tenant_id=tenant.tenant_id,
        )
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    return {
        "chain_id": result.chain_id,
        "is_valid": result.is_valid,
        "link_count": result.link_count,
        "violations": result.violations,
        "verified_at": result.verified_at.isoformat(),
    }


@router.post("/keys", status_code=status.HTTP_201_CREATED, response_model=PublicKeyResponse)
async def register_public_key(
    request: RegisterPublicKeyRequest,
    tenant: TenantDep,
    service: ProvServiceDep,
    session: Annotated[Any, Depends(get_db_session)],
) -> Any:
    """Register an Ed25519 public key for signing provenance chain links.

    The key will be stored and used to verify signatures on chain links
    that reference this key's UUID.
    """
    try:
        key_record = await service.register_public_key(
            session=session,
            tenant_id=tenant.tenant_id,
            key_label=request.key_label,
            public_key_pem=request.public_key_pem,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    return key_record

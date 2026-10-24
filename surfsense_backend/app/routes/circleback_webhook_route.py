"""
Circleback Webhook Route

This module provides a webhook endpoint for receiving meeting data from Circleback.
It processes the incoming webhook payload and saves it as a document in the specified workspace.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import SearchSourceConnector, SearchSourceConnectorType, get_async_session
from app.routes.circleback_webhook_auth import verify_circleback_signature
from app.routes.circleback_webhook_idempotency import (
    DeliveryStoreUnavailableError,
    claim_circleback_delivery,
    release_circleback_delivery,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# Pydantic models for Circleback webhook payload
class CirclebackAttendee(BaseModel):
    """Attendee model for Circleback meeting."""

    name: str | None = None
    email: str | None = None


class CirclebackActionItemAssignee(BaseModel):
    """Assignee model for action items."""

    name: str | None = None
    email: str | None = None


class CirclebackActionItem(BaseModel):
    """Action item model for Circleback meeting."""

    id: int
    title: str
    description: str = ""
    assignee: CirclebackActionItemAssignee | None = None
    status: str = "PENDING"


class CirclebackTranscriptSegment(BaseModel):
    """Transcript segment model for Circleback meeting."""

    speaker: str
    text: str
    timestamp: float


class CirclebackInsightItem(BaseModel):
    """Individual insight item."""

    insight: str | dict[str, Any]
    speaker: str | None = None
    timestamp: float | None = None


class CirclebackWebhookPayload(BaseModel):
    """
    Circleback webhook payload model.

    This model represents the data sent by Circleback when a meeting is processed.
    """

    model_config = {"populate_by_name": True}

    id: str = Field(..., description="Circleback meeting ID")

    @field_validator("id", mode="before")
    @classmethod
    def normalize_meeting_id(cls, value: object) -> object:
        """Accept legacy numeric IDs while storing a canonical string."""
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        return value

    name: str = Field(..., description="Meeting name")
    created_at: str = Field(
        ..., alias="createdAt", description="Meeting creation date in ISO format"
    )
    duration: float = Field(..., description="Meeting duration in seconds")
    url: str | None = Field(None, description="URL of the virtual meeting")
    recording_url: str | None = Field(
        None,
        alias="recordingUrl",
        description="URL of the meeting recording (valid for 24 hours)",
    )
    tags: list[str] = Field(default_factory=list, description="Meeting tags")
    ical_uid: str | None = Field(
        None, alias="icalUid", description="Unique identifier of the calendar event"
    )
    attendees: list[CirclebackAttendee] = Field(
        default_factory=list, description="Meeting attendees"
    )
    notes: str = Field("", description="Meeting notes in Markdown format")
    action_items: list[CirclebackActionItem] = Field(
        default_factory=list,
        alias="actionItems",
        description="Action items from the meeting",
    )
    transcript: list[CirclebackTranscriptSegment] = Field(
        default_factory=list, description="Meeting transcript segments"
    )
    insights: dict[str, list[CirclebackInsightItem]] = Field(
        default_factory=dict, description="Custom insights from the meeting"
    )


def format_circleback_meeting_to_markdown(payload: CirclebackWebhookPayload) -> str:
    """
    Convert Circleback webhook payload to a well-formatted Markdown document.

    Args:
        payload: The Circleback webhook payload

    Returns:
        Markdown string representation of the meeting
    """
    lines = []

    # Title
    lines.append(f"# {payload.name}")
    lines.append("")

    # Meeting metadata
    lines.append("## Meeting Details")
    lines.append("")

    # Parse and format date
    try:
        created_dt = datetime.fromisoformat(payload.created_at.replace("Z", "+00:00"))
        formatted_date = created_dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except (ValueError, AttributeError):
        formatted_date = payload.created_at

    lines.append(f"- **Date:** {formatted_date}")
    lines.append(f"- **Duration:** {int(payload.duration // 60)} minutes")

    if payload.url:
        lines.append(f"- **Meeting URL:** {payload.url}")

    if payload.tags:
        lines.append(f"- **Tags:** {', '.join(payload.tags)}")

    lines.append(
        f"- **Circleback Link:** [View on Circleback](https://app.circleback.ai/meetings/{payload.id})"
    )
    lines.append("")

    # Attendees
    if payload.attendees:
        lines.append("## Attendees")
        lines.append("")
        for attendee in payload.attendees:
            name = attendee.name or "Unknown"
            if attendee.email:
                lines.append(f"- **{name}** ({attendee.email})")
            else:
                lines.append(f"- **{name}**")
        lines.append("")

    # Notes (if provided)
    if payload.notes:
        lines.append("## Meeting Notes")
        lines.append("")
        lines.append(payload.notes)
        lines.append("")

    # Action Items
    if payload.action_items:
        lines.append("## Action Items")
        lines.append("")
        for item in payload.action_items:
            status_emoji = "✅" if item.status == "DONE" else "⬜"
            assignee_text = ""
            if item.assignee and item.assignee.name:
                assignee_text = f" (Assigned to: {item.assignee.name})"

            lines.append(f"{status_emoji} **{item.title}**{assignee_text}")
            if item.description:
                lines.append(f"   {item.description}")
            lines.append("")

    # Insights
    if payload.insights:
        lines.append("## Insights")
        lines.append("")
        for insight_name, insight_items in payload.insights.items():
            lines.append(f"### {insight_name}")
            lines.append("")
            for insight_item in insight_items:
                if isinstance(insight_item.insight, dict):
                    for key, value in insight_item.insight.items():
                        lines.append(f"- **{key}:** {value}")
                else:
                    speaker_info = (
                        f" _{insight_item.speaker}_" if insight_item.speaker else ""
                    )
                    lines.append(f"- {insight_item.insight}{speaker_info}")
            lines.append("")

    # Transcript
    if payload.transcript:
        lines.append("## Transcript")
        lines.append("")
        for segment in payload.transcript:
            total_seconds = int(segment.timestamp)

            if total_seconds >= 3600:
                hours, remainder = divmod(total_seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                timestamp_str = f"[{hours:02d}:{minutes:02d}:{seconds:02d}]"
            else:
                minutes, seconds = divmod(total_seconds, 60)
                timestamp_str = f"[{minutes:02d}:{seconds:02d}]"

            lines.append(f"**{segment.speaker}** {timestamp_str}: {segment.text}")
            lines.append("")

    return "\n".join(lines)


@router.post("/webhooks/circleback/{workspace_id}")
async def receive_circleback_webhook(
    workspace_id: int,
    payload: CirclebackWebhookPayload,
    request: Request,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Receive and process a Circleback webhook.

    This endpoint receives meeting data from Circleback and saves it as a document
    in the specified workspace. The meeting data is converted to Markdown format
    and processed asynchronously.

    Args:
        workspace_id: The ID of the workspace to save the document to
        payload: The Circleback webhook payload containing meeting data
        session: Database session for looking up the connector

    Returns:
        Success message with document details

    Note:
        This endpoint does not require authentication as it's designed to receive
        webhooks from Circleback. Signature verification can be added later for security.
    """
    claim_token: str | None = None
    enqueued = False

    try:
        raw_body = await request.body()
        signature = request.headers.get("x-signature")

        if not verify_circleback_signature(workspace_id, raw_body, signature):
            raise HTTPException(
                status_code=401,
                detail="Invalid webhook signature",
            )

        try:
            claim_token = await asyncio.to_thread(
                claim_circleback_delivery, workspace_id, raw_body
            )
        except DeliveryStoreUnavailableError:
            raise HTTPException(
                status_code=503,
                detail="Webhook delivery temporarily unavailable",
            ) from None

        if claim_token is None:
            return {
                "status": "duplicate",
                "meeting_id": payload.id,
                "workspace_id": workspace_id,
            }

        logger.info(
            f"Received Circleback webhook for meeting {payload.id} in workspace {workspace_id}"
        )

        # Look up the Circleback connector for this workspace
        connector_result = await session.execute(
            select(SearchSourceConnector.id).where(
                SearchSourceConnector.workspace_id == workspace_id,
                SearchSourceConnector.connector_type
                == SearchSourceConnectorType.CIRCLEBACK_CONNECTOR,
            )
        )
        connector_id = connector_result.scalar_one_or_none()

        if connector_id:
            logger.info(
                f"Found Circleback connector {connector_id} for workspace {workspace_id}"
            )
        else:
            logger.warning(
                f"No Circleback connector found for workspace {workspace_id}. "
                "Document will be created without connector_id."
            )

        # Convert to markdown
        markdown_content = format_circleback_meeting_to_markdown(payload)

        # Trigger async document processing
        from app.tasks.celery_tasks.document_tasks import (
            process_circleback_meeting_task,
        )

        # Prepare meeting metadata for the task
        meeting_metadata = {
            "circleback_meeting_id": payload.id,
            "meeting_name": payload.name,
            "meeting_date": payload.created_at,
            "duration_seconds": payload.duration,
            "meeting_url": payload.url,
            "tags": payload.tags,
            "attendees_count": len(payload.attendees),
            "action_items_count": len(payload.action_items),
            "has_transcript": len(payload.transcript) > 0,
        }

        # Queue the processing task
        process_circleback_meeting_task.delay(
            meeting_id=payload.id,
            meeting_name=payload.name,
            markdown_content=markdown_content,
            metadata=meeting_metadata,
            workspace_id=workspace_id,
            connector_id=connector_id,
        )

        enqueued = True

        logger.info(
            f"Queued Circleback meeting {payload.id} for processing in workspace {workspace_id}"
        )

        return {
            "status": "accepted",
            "message": f"Meeting '{payload.name}' queued for processing",
            "meeting_id": payload.id,
            "workspace_id": workspace_id,
        }

    except HTTPException:
        raise
    except Exception:
        logger.exception(
            "Circleback webhook processing failed for workspace %s",
            workspace_id,
        )
        raise HTTPException(
            status_code=500,
            detail="Internal webhook processing error",
        ) from None
    finally:
        if claim_token is not None and not enqueued:
            try:
                await asyncio.to_thread(
                    release_circleback_delivery,
                    workspace_id,
                    raw_body,
                    claim_token,
                )
            except DeliveryStoreUnavailableError:
                logger.warning("Could not release Circleback delivery claim")


@router.get("/webhooks/circleback/{workspace_id}/info")
async def get_circleback_webhook_info(
    workspace_id: int,
):
    """
    Get information about the Circleback webhook endpoint.

    This endpoint provides information about how to configure the Circleback
    webhook integration.

    Args:
        workspace_id: The ID of the workspace

    Returns:
        Webhook configuration information
    """
    from app.config import config

    # Construct the webhook URL
    base_url = getattr(config, "API_BASE_URL", "http://localhost:8000")
    webhook_url = f"{base_url}/api/v1/webhooks/circleback/{workspace_id}"

    return {
        "webhook_url": webhook_url,
        "workspace_id": workspace_id,
        "method": "POST",
        "content_type": "application/json",
        "description": "Use this URL in your Circleback automation to send meeting data to SurfSense",
        "note": "Configure this URL in Circleback Settings → Automations → Create automation → Send webhook request",
    }

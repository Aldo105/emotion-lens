"""
EmotionLens — Interviewer Notes Router

CRUD for notes/tags that the interviewer adds during a session.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.database import get_db, InterviewerNote, Session
from backend.app.models.schemas import NoteCreate, NoteResponse

router = APIRouter()


@router.post("/{session_id}", response_model=NoteResponse, status_code=201)
async def create_note(
    session_id: int,
    data: NoteCreate,
    db: AsyncSession = Depends(get_db),
):
    """Add an interviewer note to a session."""
    # Verify session exists
    result = await db.execute(select(Session).where(Session.id == session_id))
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    note = InterviewerNote(
        session_id=session_id,
        timestamp=data.timestamp,
        content=data.content,
        tag=data.tag,
    )
    db.add(note)
    await db.flush()
    await db.refresh(note)
    return note


@router.get("/{session_id}", response_model=list[NoteResponse])
async def list_notes(
    session_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get all notes for a session, ordered by timestamp."""
    result = await db.execute(
        select(InterviewerNote)
        .where(InterviewerNote.session_id == session_id)
        .order_by(InterviewerNote.timestamp.asc())
    )
    return result.scalars().all()


@router.delete("/{note_id}", status_code=204)
async def delete_note(note_id: int, db: AsyncSession = Depends(get_db)):
    """Delete a specific note."""
    result = await db.execute(
        select(InterviewerNote).where(InterviewerNote.id == note_id)
    )
    note = result.scalar_one_or_none()
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    await db.delete(note)

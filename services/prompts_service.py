"""
Service for managing prompt examples and techniques
"""
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from sqlalchemy import create_engine, text, desc
from sqlalchemy.orm import sessionmaker
from models.prompts import PromptExample, PromptTechnique

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Database connection
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5433")
DB_NAME = os.getenv("APP_DB", "app")
DB_USER = os.getenv("APP_DB_USER", "app_user")
DB_PASSWORD = os.getenv("APP_DB_PASSWORD", "app_password")

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Create engine and session factory
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_latest_prompts(
    technique_key: Optional[str] = None,
    q: Optional[str] = None,
    enabled: Optional[bool] = None
) -> List[PromptExample]:
    """
    Get latest prompts from v_prompt_examples_latest view with filters.
    
    Args:
        technique_key: Filter by technique key
        q: Search query (searches in title and technique_key)
        enabled: Filter by is_enabled flag
    
    Returns:
        List of PromptExample objects
    """
    session = SessionLocal()
    try:
        # Query from the view
        query = text("""
            SELECT * FROM app.v_prompt_examples_latest
            WHERE 1=1
            AND (:technique_key IS NULL OR technique_key = :technique_key)
            AND (:enabled IS NULL OR is_enabled = :enabled)
            AND (:q IS NULL OR title ILIKE :q_pattern OR technique_key ILIKE :q_pattern)
            ORDER BY created_at DESC
        """)
        
        q_pattern = f"%{q}%" if q else None
        result = session.execute(
            query,
            {
                "technique_key": technique_key,
                "enabled": enabled,
                "q": q,
                "q_pattern": q_pattern
            }
        )
        
        # Convert to PromptExample objects
        prompts = []
        for row in result:
            prompt = PromptExample(
                example_id=row.example_id,
                technique_key=row.technique_key,
                title=row.title,
                version=row.version,
                status=row.status,
                language=row.language,
                messages=row.messages,
                variables=row.variables,
                model_hint=row.model_hint,
                is_enabled=row.is_enabled,
                created_at=row.created_at,
                updated_at=row.updated_at
            )
            prompts.append(prompt)
        
        return prompts
    finally:
        session.close()


def create_prompt(
    technique_key: str,
    title: str,
    messages: List[Dict[str, str]],
    language: str = "en",
    variables: Optional[List[Dict[str, Any]]] = None,
    model_hint: Optional[str] = None
) -> PromptExample:
    """
    Create a new prompt (version 1).
    
    Args:
        technique_key: Technique key (must exist in prompt_techniques)
        title: Prompt title
        messages: List of message dicts
        language: Language code (default 'en')
        variables: List of variable definitions
        model_hint: Optional model recommendation
    
    Returns:
        Created PromptExample object
    """
    session = SessionLocal()
    try:
        prompt = PromptExample(
            technique_key=technique_key,
            title=title,
            version=1,
            status="active",
            language=language,
            messages=messages,
            variables=variables or [],
            model_hint=model_hint,
            is_enabled=True
        )
        
        session.add(prompt)
        session.commit()
        session.refresh(prompt)
        return prompt
    finally:
        session.close()


def get_prompt_by_id(example_id: str) -> Optional[PromptExample]:
    """
    Get a prompt by its example_id.
    
    Args:
        example_id: UUID of the prompt
    
    Returns:
        PromptExample object or None if not found
    """
    session = SessionLocal()
    try:
        prompt = session.query(PromptExample).filter(
            PromptExample.example_id == example_id
        ).first()
        
        if prompt:
            # Detach from session to avoid expiration issues
            session.expunge(prompt)
        
        return prompt
    finally:
        session.close()


def create_new_version(
    example_id: str,
    messages: Optional[List[Dict[str, str]]] = None,
    variables: Optional[List[Dict[str, Any]]] = None,
    model_hint: Optional[str] = None,
    language: Optional[str] = None,
    auto_archive_previous: bool = False
) -> Optional[PromptExample]:
    """
    Create a new version of an existing prompt.
    
    Args:
        example_id: UUID of the base prompt to clone
        messages: Optional new messages (uses base if None)
        variables: Optional new variables (uses base if None)
        model_hint: Optional new model hint (uses base if None)
        language: Optional new language (uses base if None)
        auto_archive_previous: If True, archives the previous active version
    
    Returns:
        New PromptExample object or None if base not found
    """
    session = SessionLocal()
    try:
        # Get the base prompt
        base_prompt = session.query(PromptExample).filter(
            PromptExample.example_id == example_id
        ).first()
        
        if not base_prompt:
            return None
        
        # Get the max version for this technique_key + title
        max_version = session.query(PromptExample).filter(
            PromptExample.technique_key == base_prompt.technique_key,
            PromptExample.title == base_prompt.title
        ).order_by(desc(PromptExample.version)).first()
        
        next_version = (max_version.version if max_version else 0) + 1
        
        # Optionally archive the previous active version
        if auto_archive_previous:
            previous_active = session.query(PromptExample).filter(
                PromptExample.technique_key == base_prompt.technique_key,
                PromptExample.title == base_prompt.title,
                PromptExample.status == "active"
            ).first()
            
            if previous_active:
                previous_active.status = "archived"
        
        # Create new version
        new_prompt = PromptExample(
            technique_key=base_prompt.technique_key,
            title=base_prompt.title,
            version=next_version,
            status="active",
            language=language if language is not None else base_prompt.language,
            messages=messages if messages is not None else base_prompt.messages,
            variables=variables if variables is not None else base_prompt.variables,
            model_hint=model_hint if model_hint is not None else base_prompt.model_hint,
            is_enabled=True
        )
        
        session.add(new_prompt)
        session.commit()
        session.refresh(new_prompt)
        
        # Detach from session
        session.expunge(new_prompt)
        return new_prompt
    finally:
        session.close()


def get_latest_by_technique_and_title(
    technique_key: str,
    title: str
) -> Optional[PromptExample]:
    """
    Get the latest active and enabled prompt by technique_key and title.
    
    Args:
        technique_key: Technique key
        title: Prompt title
    
    Returns:
        PromptExample object or None if not found
    """
    session = SessionLocal()
    try:
        query = text("""
            SELECT * FROM app.v_prompt_examples_latest
            WHERE technique_key = :technique_key
            AND title = :title
            LIMIT 1
        """)
        
        result = session.execute(
            query,
            {"technique_key": technique_key, "title": title}
        ).first()
        
        if not result:
            return None
        
        prompt = PromptExample(
            example_id=result.example_id,
            technique_key=result.technique_key,
            title=result.title,
            version=result.version,
            status=result.status,
            language=result.language,
            messages=result.messages,
            variables=result.variables,
            model_hint=result.model_hint,
            is_enabled=result.is_enabled,
            created_at=result.created_at,
            updated_at=result.updated_at
        )
        
        return prompt
    finally:
        session.close()


def get_prompt_by_id_and_version(
    example_id: str,
    version: Optional[int] = None
) -> Optional[PromptExample]:
    """
    Get a prompt by example_id and optionally specific version.
    If version is None, gets the prompt with that example_id (exact match).
    
    Args:
        example_id: UUID of the prompt
        version: Optional version number
    
    Returns:
        PromptExample object or None if not found
    """
    session = SessionLocal()
    try:
        if version is None:
            # Get by exact example_id
            prompt = session.query(PromptExample).filter(
                PromptExample.example_id == example_id
            ).first()
        else:
            # Get by technique_key + title + version from the base example
            base = session.query(PromptExample).filter(
                PromptExample.example_id == example_id
            ).first()
            
            if not base:
                return None
            
            prompt = session.query(PromptExample).filter(
                PromptExample.technique_key == base.technique_key,
                PromptExample.title == base.title,
                PromptExample.version == version
            ).first()
        
        if prompt:
            session.expunge(prompt)
        
        return prompt
    finally:
        session.close()


def update_prompt_status(
    example_id: str,
    is_enabled: Optional[bool] = None,
    status: Optional[str] = None
) -> Optional[PromptExample]:
    """
    Update prompt's is_enabled flag and/or status.
    
    Args:
        example_id: UUID of the prompt
        is_enabled: New enabled flag (optional)
        status: New status (optional)
    
    Returns:
        Updated PromptExample object or None if not found
    """
    session = SessionLocal()
    try:
        prompt = session.query(PromptExample).filter(
            PromptExample.example_id == example_id
        ).first()
        
        if not prompt:
            return None
        
        if is_enabled is not None:
            prompt.is_enabled = is_enabled
        
        if status is not None:
            prompt.status = status
        
        session.commit()
        session.refresh(prompt)
        session.expunge(prompt)
        
        return prompt
    finally:
        session.close()


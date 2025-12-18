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
DB_HOST = os.getenv("PGB_APP_DB_HOST")
DB_PORT = os.getenv("PGB_APP_DB_PORT")
DB_NAME = os.getenv("APP_DB", "app")
DB_USER = os.getenv("APP_DB_USER", "app_user")
DB_PASSWORD = os.getenv("APP_DB_PASSWORD", "app_password")

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Create engine and session factory
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_latest_prompts(
    technique: Optional[str] = None,
    q: Optional[str] = None,
    is_active: Optional[bool] = None
) -> List[PromptExample]:
    """
    Get latest prompts from v_prompt_examples_latest view with filters.
    
    Args:
        technique: Filter by technique ('zero_shot', 'few_shot', 'prompt_chain')
        q: Search query (searches in title and description)
        is_active: Filter by is_active flag
    
    Returns:
        List of PromptExample objects
    """
    session = SessionLocal()
    try:
        # Query from the view
        query = text("""
            SELECT * FROM app.v_prompt_examples_latest
            WHERE 1=1
            AND (:technique IS NULL OR technique = :technique)
            AND (:is_active IS NULL OR is_active = :is_active)
            AND (:q_pattern IS NULL OR title ILIKE :q_pattern OR description ILIKE :q_pattern)
            ORDER BY created_at DESC
        """)
        
        q_pattern = f"%{q}%" if q else None
        result = session.execute(
            query,
            {
                "technique": technique,
                "is_active": is_active,
                "q_pattern": q_pattern
            }
        )
        
        # Convert to PromptExample objects
        prompts = []
        for row in result:
            prompt = PromptExample(
                id=row.id,
                key=row.key,
                version=row.version,
                title=row.title,
                description=row.description,
                category=row.category,
                technique=row.technique,
                tags=row.tags or [],
                prompt_template=row.prompt_template,
                variables=row.variables or [],
                default_examples=row.default_examples,
                response_format=row.response_format,
                json_schema_template=row.json_schema_template,
                tool_config=row.tool_config,
                is_active=row.is_active,
                created_at=row.created_at,
                updated_at=row.updated_at
            )
            prompts.append(prompt)
        
        return prompts
    finally:
        session.close()


def create_prompt(
    key: str,
    title: str,
    prompt_template: List[Dict[str, str]],
    technique: str,
    description: Optional[str] = None,
    category: Optional[str] = None,
    tags: Optional[List[str]] = None,
    variables: Optional[List[Dict[str, Any]]] = None,
    default_examples: Optional[List[Dict[str, Any]]] = None,
    response_format: Optional[str] = None,
    json_schema_template: Optional[Dict[str, Any]] = None,
    tool_config: Optional[Dict[str, Any]] = None
) -> PromptExample:
    """
    Create a new prompt (version 1).
    
    Args:
        key: Internal key (unique identifier)
        title: Prompt title
        prompt_template: Messages array in LLM format [{"role": "system", "content": "..."}, ...] with Jinja variables
        technique: Technique type ('zero_shot', 'few_shot', 'prompt_chain')
        description: Optional description
        category: Optional category (e.g. 'qa', 'summarization', 'coding')
        tags: Optional list of tags
        variables: List of variable definitions
        default_examples: Optional few-shot examples (for few_shot technique)
        response_format: Response format (None, 'json_object', 'json_schema')
        json_schema_template: Optional JSON schema template (only used when response_format='json_schema')
        tool_config: Optional tool config
    
    Returns:
        Created PromptExample object
    """
    session = SessionLocal()
    try:
        prompt = PromptExample(
            key=key,
            title=title,
            version=1,
            description=description,
            category=category,
            technique=technique,
            tags=tags or [],
            prompt_template=prompt_template,
            variables=variables or [],
            default_examples=default_examples,
            response_format=response_format,
            json_schema_template=json_schema_template,
            tool_config=tool_config,
            is_active=True
        )
        
        session.add(prompt)
        session.commit()
        session.refresh(prompt)
        return prompt
    finally:
        session.close()


def get_prompt_by_id(prompt_id: str) -> Optional[PromptExample]:
    """
    Get a prompt by its id.
    
    Args:
        prompt_id: UUID of the prompt
    
    Returns:
        PromptExample object or None if not found
    """
    session = SessionLocal()
    try:
        prompt = session.query(PromptExample).filter(
            PromptExample.id == prompt_id
        ).first()
        
        if prompt:
            # Detach from session to avoid expiration issues
            session.expunge(prompt)
        
        return prompt
    finally:
        session.close()


def create_new_version(
    prompt_id: str,
    prompt_template: Optional[List[Dict[str, str]]] = None,
    variables: Optional[List[Dict[str, Any]]] = None,
    default_examples: Optional[List[Dict[str, Any]]] = None,
    response_format: Optional[str] = None,
    json_schema_template: Optional[Dict[str, Any]] = None,
    tool_config: Optional[Dict[str, Any]] = None,
    description: Optional[str] = None,
    category: Optional[str] = None,
    tags: Optional[List[str]] = None,
    auto_deactivate_previous: bool = False
) -> Optional[PromptExample]:
    """
    Create a new version of an existing prompt.
    
    Args:
        prompt_id: UUID of the base prompt to clone
        prompt_template: Optional new template (uses base if None)
        variables: Optional new variables (uses base if None)
        default_examples: Optional new default examples (uses base if None)
        response_format: Optional new response format (uses base if None)
        json_schema_template: Optional new JSON schema template (uses base if None)
        tool_config: Optional new tool config (uses base if None)
        description: Optional new description (uses base if None)
        category: Optional new category (uses base if None)
        tags: Optional new tags (uses base if None)
        auto_deactivate_previous: If True, deactivates the previous active version
    
    Returns:
        New PromptExample object or None if base not found
    """
    session = SessionLocal()
    try:
        # Get the base prompt
        base_prompt = session.query(PromptExample).filter(
            PromptExample.id == prompt_id
        ).first()
        
        if not base_prompt:
            return None
        
        # Get the max version for this key
        max_version = session.query(PromptExample).filter(
            PromptExample.key == base_prompt.key
        ).order_by(desc(PromptExample.version)).first()
        
        next_version = (max_version.version if max_version else 0) + 1
        
        # Optionally deactivate the previous active version
        if auto_deactivate_previous:
            previous_active = session.query(PromptExample).filter(
                PromptExample.key == base_prompt.key,
                PromptExample.is_active == True
            ).first()
            
            if previous_active:
                previous_active.is_active = False
        
        # Create new version
        new_prompt = PromptExample(
            key=base_prompt.key,
            title=base_prompt.title,
            version=next_version,
            description=description if description is not None else base_prompt.description,
            category=category if category is not None else base_prompt.category,
            technique=base_prompt.technique,
            tags=tags if tags is not None else base_prompt.tags,
            prompt_template=prompt_template if prompt_template is not None else base_prompt.prompt_template,
            variables=variables if variables is not None else base_prompt.variables,
            default_examples=default_examples if default_examples is not None else base_prompt.default_examples,
            response_format=response_format if response_format is not None else base_prompt.response_format,
            json_schema_template=json_schema_template if json_schema_template is not None else base_prompt.json_schema_template,
            tool_config=tool_config if tool_config is not None else base_prompt.tool_config,
            is_active=True
        )
        
        session.add(new_prompt)
        session.commit()
        session.refresh(new_prompt)
        
        # Detach from session
        session.expunge(new_prompt)
        return new_prompt
    finally:
        session.close()


def get_latest_by_key(
    key: str
) -> Optional[PromptExample]:
    """
    Get the latest active prompt by key.
    
    Args:
        key: Prompt key
    
    Returns:
        PromptExample object or None if not found
    """
    session = SessionLocal()
    try:
        query = text("""
            SELECT * FROM app.v_prompt_examples_latest
            WHERE key = :key
            LIMIT 1
        """)
        
        result = session.execute(
            query,
            {"key": key}
        ).first()
        
        if not result:
            return None
        
        prompt = PromptExample(
            id=result.id,
            key=result.key,
            version=result.version,
            title=result.title,
            description=result.description,
            category=result.category,
            technique=result.technique,
            tags=result.tags or [],
            prompt_template=result.prompt_template,
            variables=result.variables or [],
            default_examples=result.default_examples,
            response_format=result.response_format,
            json_schema_template=result.json_schema_template,
            tool_config=result.tool_config,
            is_active=result.is_active,
            created_at=result.created_at,
            updated_at=result.updated_at
        )
        
        return prompt
    finally:
        session.close()


def get_prompt_by_id_and_version(
    prompt_id: str,
    version: Optional[int] = None
) -> Optional[PromptExample]:
    """
    Get a prompt by id and optionally specific version.
    If version is None, gets the prompt with that id (exact match).
    
    Args:
        prompt_id: UUID of the prompt
        version: Optional version number
    
    Returns:
        PromptExample object or None if not found
    """
    session = SessionLocal()
    try:
        if version is None:
            # Get by exact id
            prompt = session.query(PromptExample).filter(
                PromptExample.id == prompt_id
            ).first()
        else:
            # Get by key + version from the base example
            base = session.query(PromptExample).filter(
                PromptExample.id == prompt_id
            ).first()
            
            if not base:
                return None
            
            prompt = session.query(PromptExample).filter(
                PromptExample.key == base.key,
                PromptExample.version == version
            ).first()
        
        if prompt:
            session.expunge(prompt)
        
        return prompt
    finally:
        session.close()


def update_prompt_status(
    prompt_id: str,
    is_active: Optional[bool] = None
) -> Optional[PromptExample]:
    """
    Update prompt's is_active flag.
    
    Args:
        prompt_id: UUID of the prompt
        is_active: New active flag (optional)
    
    Returns:
        Updated PromptExample object or None if not found
    """
    session = SessionLocal()
    try:
        prompt = session.query(PromptExample).filter(
            PromptExample.id == prompt_id
        ).first()
        
        if not prompt:
            return None
        
        if is_active is not None:
            prompt.is_active = is_active
        
        session.commit()
        session.refresh(prompt)
        session.expunge(prompt)
        
        return prompt
    finally:
        session.close()


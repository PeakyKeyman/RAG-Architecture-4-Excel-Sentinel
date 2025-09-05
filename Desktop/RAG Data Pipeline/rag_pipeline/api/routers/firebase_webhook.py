"""
Firebase Storage webhook integration for automatic document ingestion.
This endpoint receives notifications when files are uploaded to Firebase Storage.
"""

import json
import base64
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Request, status, BackgroundTasks
from pydantic import BaseModel, Field
from google.cloud import storage

from .document_ingestion import get_document_parser
from ...core.logging_config import get_logger
from ...parsing.document_parser import DocumentParser, DocumentTypeError


router = APIRouter(prefix="/firebase", tags=["firebase-integration"])
logger = get_logger(__name__, "firebase_webhook")


class FirebaseStorageEvent(BaseModel):
    """Firebase Storage event data model."""
    bucket: str
    name: str  # File path in bucket
    metageneration: str
    timeCreated: str
    eventType: str = Field(..., description="Event type (e.g., 'google.storage.object.finalize')")
    eventTime: str


class PubSubMessage(BaseModel):
    """Google Cloud Pub/Sub message wrapper."""
    data: str  # Base64 encoded JSON
    attributes: Dict[str, str] = Field(default_factory=dict)
    messageId: str
    publishTime: str


class WebhookRequest(BaseModel):
    """Webhook request from Firebase/Cloud Functions."""
    message: PubSubMessage


class ProcessingJobStatus(BaseModel):
    """Background job status tracking."""
    job_id: str
    status: str
    file_path: str
    org_id: Optional[str] = None
    user_id: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


# In-memory job tracking (in production, use Redis or database)
processing_jobs: Dict[str, ProcessingJobStatus] = {}


def extract_metadata_from_path(file_path: str) -> Dict[str, Any]:
    """
    Extract org_id, user_id, group_id from Firebase Storage path structure.
    
    Expected path structure:
    - private-research/{org_id}/{user_id}/{group_id}/filename.pdf
    - file-upload/{org_id}/{user_id}/filename.docx
    """
    try:
        path_parts = file_path.strip('/').split('/')
        
        if len(path_parts) < 4:
            logger.warning(f"Unexpected file path structure: {file_path}")
            return {}
        
        bucket_folder = path_parts[0]  # private-research or file-upload
        org_id = path_parts[1]
        user_id = path_parts[2]
        
        # Check if there's a group_id
        if len(path_parts) >= 5:
            group_id = path_parts[3]
            filename = '/'.join(path_parts[4:])
        else:
            group_id = "default"  # Default group
            filename = '/'.join(path_parts[3:])
        
        # Determine document category and access level based on folder
        if bucket_folder == 'private-research':
            access_level = 'confidential'
            category = 'research'
        elif bucket_folder == 'file-upload':
            access_level = 'standard'
            category = 'general'
        else:
            access_level = 'standard'
            category = 'general'
        
        return {
            'org_id': org_id,
            'user_id': user_id,
            'group_id': group_id,
            'filename': filename,
            'access_level': access_level,
            'category': category,
            'bucket_folder': bucket_folder
        }
        
    except Exception as e:
        logger.error(f"Failed to extract metadata from path {file_path}: {str(e)}")
        return {}


async def process_uploaded_document(
    bucket_name: str,
    file_path: str, 
    metadata: Dict[str, Any],
    job_id: str
) -> None:
    """
    Background task to process uploaded document.
    
    Args:
        bucket_name: Firebase Storage bucket name
        file_path: Path to file in bucket
        metadata: Extracted metadata (org_id, user_id, etc.)
        job_id: Unique job identifier for tracking
    """
    
    # Update job status
    if job_id in processing_jobs:
        processing_jobs[job_id].status = "processing"
    
    try:
        logger.info(
            f"Starting document processing for Firebase upload",
            extra={
                "job_id": job_id,
                "bucket": bucket_name,
                "file_path": file_path,
                "org_id": metadata.get('org_id')
            }
        )
        
        # Initialize document parser
        parser = get_document_parser()
        
        # Construct GCS URI
        gcs_uri = f"gs://{bucket_name}/{file_path}"
        
        # Parse document with extracted metadata
        document = parser.parse_document(
            file_path=gcs_uri,
            user_id=metadata['user_id'],
            group_id=metadata['group_id'],
            org_id=metadata['org_id'],
            document_metadata={
                'access_level': metadata['access_level'],
                'category': metadata['category'],
                'source': 'firebase_upload',
                'bucket_folder': metadata['bucket_folder'],
                'original_filename': metadata['filename']
            }
        )
        
        # TODO: Chunk and upsert to vector store
        # This would integrate with the hierarchical chunker and vector store
        # For now, we'll just log success
        
        # Update job status
        if job_id in processing_jobs:
            processing_jobs[job_id].status = "completed"
            processing_jobs[job_id].completed_at = datetime.now()
        
        logger.info(
            f"Successfully processed Firebase upload",
            extra={
                "job_id": job_id,
                "document_id": document.document_id,
                "org_id": metadata.get('org_id'),
                "file_path": file_path
            }
        )
        
    except DocumentTypeError as e:
        error_msg = f"Unsupported document type: {str(e)}"
        logger.warning(error_msg, extra={"job_id": job_id, "file_path": file_path})
        
        if job_id in processing_jobs:
            processing_jobs[job_id].status = "failed"
            processing_jobs[job_id].error_message = error_msg
            processing_jobs[job_id].completed_at = datetime.now()
            
    except Exception as e:
        error_msg = f"Failed to process document: {str(e)}"
        logger.error(error_msg, exc_info=True, 
                    extra={"job_id": job_id, "file_path": file_path})
        
        if job_id in processing_jobs:
            processing_jobs[job_id].status = "failed"
            processing_jobs[job_id].error_message = error_msg
            processing_jobs[job_id].completed_at = datetime.now()


@router.post("/storage-upload")
async def handle_firebase_storage_upload(
    request: WebhookRequest,
    background_tasks: BackgroundTasks,
    http_request: Request
):
    """
    Handle Firebase Storage upload notifications via Pub/Sub.
    
    This endpoint should be called by a Cloud Function that triggers
    on Firebase Storage object creation events.
    
    Example Cloud Function trigger:
    ```python
    @functions_framework.cloud_event
    def process_upload(cloud_event):
        # Extract file info from cloud_event
        requests.post("https://your-rag-api.com/firebase/storage-upload", 
                     json={"message": cloud_event.data})
    ```
    """
    request_id = getattr(http_request.state, 'request_id', 'unknown')
    
    try:
        # Decode Pub/Sub message
        message_data = base64.b64decode(request.message.data).decode('utf-8')
        event_data = json.loads(message_data)
        
        # Parse Firebase Storage event
        storage_event = FirebaseStorageEvent(**event_data)
        
        # Only process object finalize events (file uploads)
        if storage_event.eventType != 'google.storage.object.finalize':
            logger.info(f"Ignoring event type: {storage_event.eventType}")
            return {"success": True, "message": f"Ignored event type: {storage_event.eventType}"}
        
        # Extract metadata from file path
        metadata = extract_metadata_from_path(storage_event.name)
        
        if not metadata or not all(k in metadata for k in ['org_id', 'user_id', 'group_id']):
            logger.warning(f"Could not extract required metadata from path: {storage_event.name}")
            return {"success": False, "message": "Invalid file path structure"}
        
        # Check if file type is supported
        supported_extensions = {'.pdf', '.docx', '.pptx'}
        file_extension = '.' + storage_event.name.split('.')[-1].lower()
        
        if file_extension not in supported_extensions:
            logger.info(f"Unsupported file type {file_extension}, ignoring: {storage_event.name}")
            return {"success": True, "message": f"Unsupported file type: {file_extension}"}
        
        # Create processing job
        job_id = f"{storage_event.bucket}_{storage_event.name}_{storage_event.timeCreated}".replace('/', '_')
        
        processing_jobs[job_id] = ProcessingJobStatus(
            job_id=job_id,
            status="queued",
            file_path=storage_event.name,
            org_id=metadata.get('org_id'),
            user_id=metadata.get('user_id'),
            created_at=datetime.now()
        )
        
        # Queue background processing
        background_tasks.add_task(
            process_uploaded_document,
            storage_event.bucket,
            storage_event.name,
            metadata,
            job_id
        )
        
        logger.info(
            f"Queued Firebase upload for processing",
            extra={
                "request_id": request_id,
                "job_id": job_id,
                "bucket": storage_event.bucket,
                "file_path": storage_event.name,
                "org_id": metadata.get('org_id')
            }
        )
        
        return {
            "success": True,
            "message": "Document queued for processing",
            "job_id": job_id,
            "file_path": storage_event.name,
            "org_id": metadata.get('org_id')
        }
        
    except Exception as e:
        logger.error(f"Firebase webhook processing failed: {str(e)}", 
                    exc_info=True, extra={"request_id": request_id})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Webhook processing failed: {str(e)}"
        )


@router.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of a document processing job."""
    if job_id not in processing_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )
    
    job = processing_jobs[job_id]
    return {
        "job_id": job.job_id,
        "status": job.status,
        "file_path": job.file_path,
        "org_id": job.org_id,
        "user_id": job.user_id,
        "created_at": job.created_at.isoformat(),
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "error_message": job.error_message
    }


@router.get("/jobs")
async def list_jobs(
    org_id: Optional[str] = None,
    user_id: Optional[str] = None,
    status_filter: Optional[str] = None
):
    """List processing jobs with optional filtering."""
    filtered_jobs = []
    
    for job in processing_jobs.values():
        # Apply filters
        if org_id and job.org_id != org_id:
            continue
        if user_id and job.user_id != user_id:
            continue
        if status_filter and job.status != status_filter:
            continue
            
        filtered_jobs.append({
            "job_id": job.job_id,
            "status": job.status,
            "file_path": job.file_path,
            "org_id": job.org_id,
            "user_id": job.user_id,
            "created_at": job.created_at.isoformat(),
            "completed_at": job.completed_at.isoformat() if job.completed_at else None
        })
    
    return {
        "jobs": filtered_jobs,
        "total_count": len(filtered_jobs)
    }
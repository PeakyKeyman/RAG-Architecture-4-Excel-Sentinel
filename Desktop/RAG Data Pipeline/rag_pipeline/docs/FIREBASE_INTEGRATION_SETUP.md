# Firebase Integration Setup Guide

This guide explains how to integrate your RAG Data Pipeline with Firebase Storage for automatic document ingestion.

## Architecture Overview

```
Firebase Storage Upload → Cloud Function → RAG Pipeline → Vertex AI Vector Search
                                       ↓
                              Multi-tenant Security
                              (org_id/user_id/group_id)
```

## Prerequisites

1. **Firebase Project**: Your existing Firebase project with Storage enabled
2. **Google Cloud Project**: Same project with Vertex AI enabled  
3. **RAG Pipeline**: Deployed and accessible via HTTPS
4. **Service Accounts**: Proper authentication configured

## Step 1: Install New Dependencies

Add the new document parsing dependencies to your environment:

```bash
pip install PyPDF2>=3.0.0 pdfplumber>=0.10.0 python-docx>=1.1.0 python-pptx>=0.6.22 pandas>=2.0.0
```

## Step 2: Configure Storage Bucket Structure

Organize your Firebase Storage with the following structure for multi-tenant security:

```
your-firebase-bucket/
├── private-research/           # Confidential executive documents
│   └── {org_id}/
│       └── {user_id}/
│           └── {group_id}/
│               ├── strategy_deck.pptx
│               ├── financial_report.pdf
│               └── board_minutes.docx
├── file-upload/               # Standard business documents  
│   └── {org_id}/
│       └── {user_id}/
│           ├── policy_doc.pdf
│           ├── market_research.docx
│           └── competitive_analysis.pptx
└── public-results/           # Non-sensitive outputs (existing)
```

**Path Structure Rules:**
- `private-research/{org_id}/{user_id}/{group_id}/filename.ext` → Access Level: Confidential
- `file-upload/{org_id}/{user_id}/filename.ext` → Access Level: Standard  
- Files must be PDF, DOCX, or PPTX format
- `org_id`, `user_id`, and `group_id` are extracted from the path for security

## Step 3: Create Cloud Function Trigger

Create a Cloud Function that triggers on Firebase Storage uploads and calls your RAG pipeline.

### 3.1 Create `functions/main.py`:

```python
import functions_framework
import requests
import json
from google.cloud import storage

# Your RAG pipeline URL
RAG_PIPELINE_URL = "https://your-rag-api.com"  # Replace with your actual URL

@functions_framework.cloud_event
def process_storage_upload(cloud_event):
    """Triggered by Firebase Storage file upload."""
    
    # Extract file information
    data = cloud_event.data
    bucket_name = data.get('bucket')
    file_name = data.get('name')
    event_type = data.get('eventType')
    
    print(f"Processing file: {file_name} in bucket: {bucket_name}")
    
    # Only process file creation events
    if event_type != 'google.storage.object.finalize':
        print(f"Ignoring event type: {event_type}")
        return
    
    # Only process supported file types
    supported_extensions = ('.pdf', '.docx', '.pptx')
    if not file_name.lower().endswith(supported_extensions):
        print(f"Ignoring unsupported file type: {file_name}")
        return
    
    # Only process files in the correct directories
    if not (file_name.startswith('private-research/') or file_name.startswith('file-upload/')):
        print(f"Ignoring file outside monitored directories: {file_name}")
        return
    
    try:
        # Call RAG pipeline webhook
        webhook_payload = {
            "message": {
                "data": json.dumps(data),
                "messageId": cloud_event.get('id', ''),
                "publishTime": data.get('timeCreated', ''),
                "attributes": {}
            }
        }
        
        response = requests.post(
            f"{RAG_PIPELINE_URL}/firebase/storage-upload",
            json=webhook_payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Successfully queued for processing: {result}")
        else:
            print(f"Failed to process upload: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"Error calling RAG pipeline: {str(e)}")
        raise
```

### 3.2 Create `functions/requirements.txt`:

```txt
functions-framework>=3.0.0
requests>=2.31.0
google-cloud-storage>=2.10.0
```

### 3.3 Deploy Cloud Function:

```bash
# Navigate to functions directory
cd functions/

# Deploy with Storage trigger
gcloud functions deploy process-storage-upload \
    --gen2 \
    --runtime=python311 \
    --source=. \
    --entry-point=process_storage_upload \
    --trigger-event-filters="type=google.cloud.storage.object.v1.finalized" \
    --trigger-event-filters="bucket=your-firebase-bucket-name" \
    --region=us-central1
```

## Step 4: Configure Firebase Storage Rules

Update your `storage.rules` to enforce the path structure:

```javascript
rules_version = '2';
service firebase.storage {
  match /b/{bucket}/o {
    // Private research documents - strict access control
    match /private-research/{org_id}/{user_id}/{group_id}/{filename} {
      allow read, write: if request.auth != null 
        && request.auth.token.org_id == org_id
        && request.auth.token.uid == user_id
        && hasGroupAccess(group_id);
    }
    
    // Standard file uploads - org and user level access
    match /file-upload/{org_id}/{user_id}/{filename} {
      allow read, write: if request.auth != null
        && request.auth.token.org_id == org_id
        && request.auth.token.uid == user_id;
    }
    
    // Public results (existing)
    match /public-results/{filename} {
      allow read: if request.auth != null;
      allow write: if false; // Only backend can write
    }
    
    // Helper function to check group access
    function hasGroupAccess(group_id) {
      return group_id in request.auth.token.group_ids;
    }
  }
}
```

## Step 5: Start Your RAG Pipeline

Ensure your RAG pipeline is running with the new endpoints:

```bash
# Install new dependencies
pip install -r requirements.txt

# Start the API server
uvicorn rag_pipeline.api.main:app --host 0.0.0.0 --port 8000
```

### New Endpoints Available:

- `POST /documents/ingest` - Manual document ingestion
- `POST /firebase/storage-upload` - Firebase webhook endpoint
- `GET /firebase/job/{job_id}` - Check processing status
- `GET /firebase/jobs` - List processing jobs
- `GET /documents/list` - List user documents with security filtering
- `DELETE /documents/{document_id}` - Delete documents with security validation

## Step 6: Configure Authentication

### 6.1 Update your Firebase Authentication to include custom claims:

```javascript
// In your Firebase Cloud Function for user creation
const admin = require('firebase-admin');

exports.setCustomUserClaims = functions.auth.user().onCreate(async (user) => {
  // Set custom claims based on your user data
  const customClaims = {
    org_id: user.customClaims?.org_id || 'default_org',
    group_ids: user.customClaims?.group_ids || ['default_group'],
    access_level: user.customClaims?.access_level || 'standard'
  };
  
  await admin.auth().setCustomUserClaims(user.uid, customClaims);
});
```

### 6.2 Update your environment variables:

```bash
# Google Cloud / Vertex AI
export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
export GEMINI_API_KEY="your-gemini-api-key"
export COHERE_API_KEY="your-cohere-api-key"
export LANGSMITH_API_KEY="your-langsmith-api-key"

# RAG Pipeline Configuration
export RAG_API_KEY="your-secure-api-key"
export LOG_LEVEL="INFO"
```

## Step 7: Test the Integration

### 7.1 Upload a test document:

Upload a PDF to: `private-research/test_org/test_user/test_group/strategy_doc.pdf`

### 7.2 Check processing status:

```bash
# Check if document was processed
curl -X GET "https://your-rag-api.com/firebase/jobs?org_id=test_org"

# Check specific job status  
curl -X GET "https://your-rag-api.com/firebase/job/{job_id}"
```

### 7.3 Query the processed document:

```bash
curl -X POST "https://your-rag-api.com/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is our strategic priority for Q4?",
    "org_id": "test_org",
    "user_id": "test_user",
    "top_k": 5
  }'
```

## Security Features

### Multi-Tenant Isolation:
- **Org-Level**: Separate Vertex AI Vector Search indexes per organization
- **User-Level**: Metadata filtering within org indexes  
- **Group-Level**: Additional access control within organizations
- **Access-Level**: Document classification (public, standard, confidential, restricted)

### Document Classification:
The system automatically classifies documents based on filename and content:
- **Financial**: Revenue reports, P&L statements, balance sheets
- **Strategic**: Vision/mission documents, OKRs, strategic plans
- **Board/Investor**: Board presentations, investor decks
- **Policy/Compliance**: Governance, risk, compliance documents
- **Market Research**: Industry analysis, competitive intelligence
- **General**: Unclassified documents

## Troubleshooting

### Common Issues:

1. **"Unsupported file type"**: Only PDF, DOCX, and PPTX files are supported
2. **"Invalid file path structure"**: Ensure files follow the org_id/user_id/group_id structure
3. **"Vector store initialization failed"**: Check Vertex AI API is enabled and credentials are correct
4. **"Permission denied"**: Verify service account has proper roles

### Debug Mode:

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
uvicorn rag_pipeline.api.main:app --reload
```

### Health Checks:

```bash
# Check system health
curl https://your-rag-api.com/health

# Check vector store health
curl https://your-rag-api.com/vector-store/stats
```

## Production Deployment

### Scaling Considerations:

1. **Vector Indexes**: Each organization gets its own Vertex AI index
2. **Compute**: Scale API server based on document processing volume
3. **Storage**: Monitor Google Cloud Storage costs for vector data
4. **Rate Limiting**: Adjust API rate limits based on organization size

### Monitoring:

- Monitor Cloud Function execution for upload processing
- Track vector store health and index statistics  
- Monitor API response times and error rates
- Set up alerts for failed document processing

This completes the Firebase integration setup. Your RAG pipeline will now automatically process executive documents as they're uploaded to Firebase Storage, with full multi-tenant security and intelligent document classification.
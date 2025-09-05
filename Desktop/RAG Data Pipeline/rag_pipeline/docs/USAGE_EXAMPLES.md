# RAG Pipeline Usage Examples

## Quick Start

### 1. Installation and Setup

```bash
# Clone the repository
git clone <repository-url>
cd rag-data-pipeline

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GEMINI_API_KEY="your-gemini-api-key"
export COHERE_API_KEY="your-cohere-api-key"
export LANGSMITH_API_KEY="your-langsmith-api-key"

# Start the API server
uvicorn rag_pipeline.api.main:app --host 0.0.0.0 --port 8000
```

### 2. Basic Query Example

```python
import requests

# Simple query
response = requests.post("http://localhost:8000/query", json={
    "query": "What are the benefits of machine learning?",
    "top_k": 5
})

result = response.json()
print(f"Answer: {result['answer']}")
print(f"Sources: {len(result['sources'])}")
```

### 3. Document Ingestion

```python
import requests

# Add documents to knowledge base
documents = [
    {
        "document_id": "ml_intro_001",
        "content": """
        Machine learning is a method of data analysis that automates analytical 
        model building. It is a branch of artificial intelligence based on the 
        idea that systems can learn from data, identify patterns and make 
        decisions with minimal human intervention.
        """,
        "metadata": {
            "source": "ml_introduction.pdf",
            "type": "educational",
            "category": "machine_learning"
        }
    },
    {
        "document_id": "dl_overview_001", 
        "content": """
        Deep learning is part of a broader family of machine learning methods 
        based on artificial neural networks with representation learning. 
        Learning can be supervised, semi-supervised or unsupervised.
        """,
        "metadata": {
            "source": "deep_learning_guide.pdf",
            "type": "technical",
            "category": "deep_learning"
        }
    }
]

response = requests.post("http://localhost:8000/documents", json={
    "documents": documents
})

print(f"Processed {response.json()['processed_documents']} documents")
```

## Advanced Usage Examples

### 1. Hybrid Search with Filters

```python
import requests

# Advanced search with filtering
response = requests.post("http://localhost:8000/search/hybrid", json={
    "query": "neural network architectures for computer vision",
    "top_k": 10,
    "enable_hyde": True,
    "rerank": True,
    "filters": {
        "category": "deep_learning",
        "type": "technical"
    }
})

results = response.json()
for i, result in enumerate(results['results']):
    print(f"{i+1}. Score: {result['score']:.3f}")
    print(f"   Content: {result['content'][:100]}...")
    print(f"   Source: {result['metadata']['source']}")
    print()
```

### 2. Streaming Query Processing

```python
import asyncio
import websockets
import json

async def stream_query():
    uri = "ws://localhost:8000/stream"
    
    async with websockets.connect(uri) as websocket:
        # Send query
        query_data = {
            "query": "Explain the transformer architecture in detail",
            "stream": True,
            "top_k": 8
        }
        await websocket.send(json.dumps(query_data))
        
        # Receive streaming response
        full_response = ""
        async for message in websocket:
            data = json.loads(message)
            
            if data.get("type") == "chunk":
                full_response += data["chunk"]
                print(data["chunk"], end="", flush=True)
            elif data.get("type") == "complete":
                print("\n\nSources:")
                for source in data["sources"]:
                    print(f"- {source['metadata']['source']}")
                break

asyncio.run(stream_query())
```

### 3. Batch Evaluation

```python
import requests

# Prepare evaluation dataset
evaluation_data = [
    {
        "question": "What is supervised learning?",
        "ground_truth": "Supervised learning is a machine learning paradigm where algorithms learn from labeled training data to make predictions on new, unseen data.",
        "contexts": [
            "Supervised learning uses labeled examples to train models",
            "In supervised learning, the algorithm learns from input-output pairs"
        ]
    },
    {
        "question": "How do neural networks work?",
        "ground_truth": "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes that process information using mathematical operations.",
        "contexts": [
            "Neural networks contain layers of interconnected neurons",
            "Each neuron applies a mathematical function to its inputs"
        ]
    }
]

# Run evaluation
response = requests.post("http://localhost:8000/evaluate", json={
    "evaluation_data": evaluation_data,
    "metrics": ["faithfulness", "answer_relevancy", "context_relevancy"]
})

results = response.json()
print(f"Evaluation Results:")
print(f"Faithfulness: {results['metrics']['faithfulness']:.3f}")
print(f"Answer Relevancy: {results['metrics']['answer_relevancy']:.3f}")
print(f"Context Relevancy: {results['metrics']['context_relevancy']:.3f}")
```

### 4. Custom Configuration

```python
import yaml
from rag_pipeline.core.config import Settings

# Load custom configuration
with open('custom_config.yaml', 'r') as f:
    config_data = yaml.safe_load(f)

# Override specific settings
config_data['chunking']['child_chunk_size'] = 300
config_data['search']['default_top_k'] = 15

# Create settings instance
settings = Settings(**config_data)

# Use in your application
from rag_pipeline.inference.pipeline import RAGPipeline

pipeline = RAGPipeline(settings=settings)
```

## Integration Examples

### 1. Flask Integration

```python
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    question = data.get('question')
    
    # Forward to RAG pipeline
    response = requests.post("http://localhost:8000/query", json={
        "query": question,
        "top_k": 5
    })
    
    if response.status_code == 200:
        result = response.json()
        return jsonify({
            "answer": result['answer'],
            "confidence": max([s['score'] for s in result['sources']]),
            "source_count": len(result['sources'])
        })
    else:
        return jsonify({"error": "Query processing failed"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

### 2. Gradio Interface

```python
import gradio as gr
import requests

def query_rag(question, top_k=5):
    response = requests.post("http://localhost:8000/query", json={
        "query": question,
        "top_k": top_k,
        "include_debug": True
    })
    
    if response.status_code == 200:
        result = response.json()
        
        # Format sources
        sources_text = "\n".join([
            f"**Source {i+1}** (Score: {s['score']:.3f})\n{s['content'][:200]}...\n"
            for i, s in enumerate(result['sources'])
        ])
        
        return result['answer'], sources_text
    else:
        return "Error processing query", ""

# Create Gradio interface
demo = gr.Interface(
    fn=query_rag,
    inputs=[
        gr.Textbox(label="Question", placeholder="Ask anything..."),
        gr.Slider(1, 20, value=5, label="Number of Sources")
    ],
    outputs=[
        gr.Textbox(label="Answer", lines=5),
        gr.Markdown(label="Sources")
    ],
    title="RAG Pipeline Demo",
    description="Ask questions and get AI-powered answers with sources."
)

demo.launch()
```

### 3. Jupyter Notebook Usage

```python
# Install in Jupyter
!pip install requests ipywidgets

import requests
import ipywidgets as widgets
from IPython.display import display, Markdown

class RAGInterface:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        
    def query(self, question, top_k=5):
        response = requests.post(f"{self.base_url}/query", json={
            "query": question,
            "top_k": top_k
        })
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.text}
    
    def display_result(self, result):
        if "error" in result:
            print(f"Error: {result['error']}")
            return
            
        # Display answer
        display(Markdown(f"**Answer:** {result['answer']}"))
        
        # Display sources
        display(Markdown("**Sources:**"))
        for i, source in enumerate(result['sources']):
            display(Markdown(f"{i+1}. **Score:** {source['score']:.3f}"))
            display(Markdown(f"   **Content:** {source['content'][:200]}..."))
            display(Markdown(f"   **Source:** {source['metadata']['source']}"))

# Usage
rag = RAGInterface()

# Interactive widget
question_widget = widgets.Text(
    placeholder='Enter your question...',
    description='Question:',
    style={'description_width': 'initial'},
    layout={'width': '500px'}
)

def on_submit(sender):
    result = rag.query(sender.value)
    rag.display_result(result)

question_widget.on_submit(on_submit)
display(question_widget)
```

## Performance Monitoring

### 1. Health Check Script

```python
import requests
import time

def check_system_health():
    try:
        response = requests.get("http://localhost:8000/health")
        health_data = response.json()
        
        print(f"System Status: {health_data['status']}")
        print("Component Status:")
        for component, status in health_data['components'].items():
            print(f"  {component}: {status}")
        print(f"Uptime: {health_data['uptime_seconds']} seconds")
        
        return health_data['status'] == 'healthy'
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

# Monitor continuously
while True:
    is_healthy = check_system_health()
    if not is_healthy:
        print("System is unhealthy!")
    time.sleep(30)  # Check every 30 seconds
```

### 2. Performance Benchmarking

```python
import requests
import time
import statistics

def benchmark_queries(queries, num_runs=10):
    results = []
    
    for query in queries:
        times = []
        
        for _ in range(num_runs):
            start_time = time.time()
            
            response = requests.post("http://localhost:8000/query", json={
                "query": query,
                "top_k": 5
            })
            
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to milliseconds
        
        results.append({
            "query": query,
            "avg_time_ms": statistics.mean(times),
            "min_time_ms": min(times),
            "max_time_ms": max(times),
            "std_dev_ms": statistics.stdev(times)
        })
    
    return results

# Run benchmark
test_queries = [
    "What is machine learning?",
    "Explain deep learning neural networks",
    "How does natural language processing work?",
    "What are the applications of computer vision?"
]

benchmark_results = benchmark_queries(test_queries)
for result in benchmark_results:
    print(f"Query: {result['query'][:50]}...")
    print(f"  Avg: {result['avg_time_ms']:.1f}ms")
    print(f"  Range: {result['min_time_ms']:.1f}ms - {result['max_time_ms']:.1f}ms")
    print()
```
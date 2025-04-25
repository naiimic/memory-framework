# Memory Framework

A modular and extensible framework for implementing hierarchical memory systems for AI agents. This framework simulates human-like memory processes with working memory, short-term memory, and long-term memory components.

## 🧠 Architecture Overview

This framework implements a cognitive architecture inspired by human memory systems:

- **Working Memory**: Short-term, limited capacity storage for immediate processing
- **Short-Term Memory**: Medium-term storage with vector embeddings for similarity retrieval
- **Long-term Memory**: Permanent storage with semantic organization and efficient retrieval

Memory flows through the system in a hierarchical manner, with summarization and forgetting mechanisms simulating natural cognitive processes.

## 🔄 Memory Flow and Mechanisms

### Memory Framework Schematic

```
                                      MEMORY FRAMEWORK SCHEMATIC
                                      -------------------------

+---------------------+      +----------------------+      +----------------------+
|                     |      |                      |      |                      |
|   WORKING MEMORY    |      |   SHORT-TERM MEMORY  |      |   LONG-TERM MEMORY   |
|                     |      |                      |      |                      |
| +---------------+   |      | +----------------+   |      | +----------------+   |
| | Simple FIFO   |   |      | | Vector-based   |   |      | | Vector-based   |   |
| | No embeddings |   |      | | Embeddings     |   |      | | Embeddings     |   |
| | Capacity: Low |   |      | | Capacity: Med  |   |      | | Capacity: High |   |
| +---------------+   |      | +----------------+   |      | +----------------+   |
|                     |      |                      |      |                      |
+----------+----------+      +-----------+----------+      +-----------+----------+
           |                             |                             |
           | New Information             | When capacity reached       | When capacity reached
           | enters here                 | or relevance triggered      | or periodically
           ↓                             ↓                             ↓
+----------+----------+      +-----------+----------+      +-----------+----------+
|                     |      |                      |      |                      |
|    DIRECT STORE     |      |   DIRECT TRANSFER    |      |      CLUSTERING      |
|                     |      |                      |      |                      |
| Raw text storage    |      | Oldest memory moved  |      | Groups similar       |
| FIFO queue          |      | to short-term memory |      | memories using       |
| No processing       |      | without summarizing  |      | DBSCAN algorithm     |
|                     |      |                      |      |                      |
+----------+----------+      +-----------+----------+      +-----------+----------+
           |                             |                             |
           | When capacity               | After transfer              | After clustering
           | is reached                  |                             |
           ↓                             ↓                             ↓
+-----------------------+    +-----------+----------+      +-----------+----------+
|                       |    |                      |      |                      |
| MOVE OLDEST MEMORY    |    |  EMBEDDING & QUERY   |      |     SUMMARIZATION    |
|                       |    |                      |      |                      |
| Transfer oldest       |    | Convert to vectors   |      | Summarize each       |
| memory to STM         +--->+ for retrieval with   |      | cluster before       |
| without summarizing   |    | forgetting mechanism +----->+ moving to LTM with   |
|                       |    |                      |      | forgetting mechanism |
+-----------------------+    +----------------------+      +----------------------+

                                          ↓

                             +-----------------------+
                             |                       |
                             |  MEMORY RETRIEVAL     |
                             |                       |
                             | 1. Query converted    |
                             |    to embedding       |
                             | 2. Similarity search  |
                             | 3. Most relevant      |
                             |    memories returned  |
                             |                       |
                             +-----------------------+
```

### Memory Types

1. **Working Memory (WM)**
   - Implemented as a simple FIFO (First-In-First-Out) queue with fixed capacity
   - No vector embeddings, just direct storage of text strings
   - When capacity is reached, oldest memory is moved to Short-Term Memory without summarization
   - Primary purpose: Immediate context and recent information processing

2. **Short-Term Memory (STM)**
   - Implemented using vector embeddings for semantic similarity search
   - Limited capacity (larger than WM but smaller than LTM)
   - When capacity is reached, content is clustered, summarized, and moved to Long-Term Memory
   - Primary purpose: Recent context retrieval and filtering information before long-term storage

3. **Long-Term Memory (LTM)**
   - Highest capacity, persistent storage using vector embeddings
   - Organized using semantic clustering for efficient retrieval
   - Implements forgetting mechanisms to manage redundant information
   - Primary purpose: Permanent knowledge storage with semantic retrieval capability

### Key Processes

#### Memory Flow
1. New information enters Working Memory
2. When WM reaches capacity, the oldest memory is moved to Short-Term Memory (without summarization)
3. When STM reaches capacity, content is clustered, summarized by cluster, and moved to Long-Term Memory
4. At each stage, forgetting mechanisms manage memory retention

#### Forgetting Mechanism
The system implements a similarity-based forgetting algorithm:
- When new memories are added to STM or LTM, they are compared to existing memories
- If similarity exceeds a threshold (default: 0.9), the older similar memories are removed
- This prevents redundant storage while preserving unique information

#### Clustering
Long-term memory implements semantic clustering:
- Uses algorithms like DBSCAN to group semantically similar memories
- Improves retrieval efficiency by organizing memories by topic
- Helps with summarization by identifying related information

#### Memory Retrieval
When the system needs to retrieve information:
1. A query is converted to the same vector embedding format
2. Similarity search is performed across all memory stores
3. Most relevant memories from each store are returned based on similarity scores
4. Content from Working Memory is always included in the context

## 📦 Project Structure

```
memory/
├── memory_architecture/   # Core memory framework implementation
│   ├── manager/           # Memory management components
│   └── module/            # Memory module implementations
├── utils/                 # Utility functions for memory operations
├── examples/              # Example scripts demonstrating the framework
├── data/                  # Sample text data for testing
├── requirements.txt       # Python dependencies
├── environment.yaml       # Conda environment specification
├── dependency_test.py     # Script to verify dependencies work
└── run_examples.py        # Script to run the examples
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+ 
- OpenAI API key (for LLM-based memory operations)
- Sufficient disk space for dependencies and embedding models

### Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd memory
   ```

2. Set up your environment:

   **Using venv (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

   **Using conda:**
   ```bash
   conda env create -f environment.yaml -n memory
   conda activate memory
   ```

3. Configure your OpenAI API key:
   
   Create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. Verify your setup:
   ```bash
   python run_examples.py check
   ```

## 🧪 Examples

The framework includes several example applications to demonstrate its capabilities:

### Story Memory Example

Demonstrates how text from a story flows through the memory system with summarization.

```bash
python run_examples.py story
```

### Interactive Memory Demo

An interactive CLI where you can input text and see how the memory system processes and retrieves information.

```bash
python run_examples.py interactive
```

### Running Tests

To ensure your setup is working correctly:

```bash
python run_examples.py test
```

## 🛠️ Core Components

### Memory Modules

- **MemoryStore**: Basic memory storage with FIFO operations
- **EmbeddingMemory**: Vector-based memory for semantic similarity retrieval

### Memory Manager

The `ChunkedMemory` manager coordinates the flow of information between memory modules, handling:

- Memory insertion and updating
- Summarization of memories using LLMs
- Memory retrieval based on relevance
- Memory organization and forgetting

### Encoders

- Uses sentence transformers to create vector embeddings
- Supports multiple encoding models for different memory types
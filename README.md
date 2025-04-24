# Memory Framework

A modular and extensible framework for implementing hierarchical memory systems for AI agents. This framework simulates human-like memory processes with working memory, recent memory, and long-term memory components.

## 🧠 Architecture Overview

This framework implements a cognitive architecture inspired by human memory systems:

- **Working Memory**: Short-term, limited capacity storage for immediate processing
- **Recent Memory**: Medium-term storage with vector embeddings for similarity retrieval
- **Long-term Memory**: Permanent storage with semantic organization and efficient retrieval

Memory flows through the system in a hierarchical manner, with summarization and forgetting mechanisms simulating natural cognitive processes.

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

## 🤝 Contributing

Contributions to improve the memory framework are welcome! Please feel free to submit issues or pull requests.

## 📄 License

[License information here]

## 📚 References

- Concepts based on cognitive science research on human memory systems
- Implements principles from the Systems for Adaptive Frameworks approach 
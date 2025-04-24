# Examples

This directory contains two simple examples to demonstrate how the Summarize-and-Forget memory framework works:

## 1. Basic Story Memory

`story_memory.py` demonstrates how the memory framework processes a short story, showing:
- How information flows through the three memory tiers
- How clustering and summarization work in practice
- How memory retrieval functions when a question is asked about story content

### Running the Example

```bash
python story_memory.py
```

## 2. Interactive Memory Demo

`interactive_memory.py` provides an interactive shell where you can:
- Add new memories through text input
- Force summarization to see how memories are clustered and condensed
- Query the memory system to retrieve related information
- Observe the memory system's state at each step

### Running the Example

```bash
python interactive_memory.py
``` 
#!/usr/bin/env python3
"""
Story Memory Example

This example demonstrates how the memory framework
processes text from a story, showing how memories flow through the system
and how summarization creates condensed representations.
"""

import os
import sys
import json
import logging
from dotenv import load_dotenv

# Add the parent directory to the path so we can import the memory_architecture module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_architecture.module.default_modules import EmbeddingMemory, MemoryStore
from memory_architecture.manager.default_manager import ChunkedMemory
from utils.encoder_utils import EncoderManager, SentenceTransformerEncoder
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Check if OpenAI API key is available
if not os.environ.get("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY not found in environment variables.")
    print("Please set your OpenAI API key in a .env file or as an environment variable.")
    sys.exit(1)

def main():
    print("\n🧠 Initializing Memory System...\n")
    
    # Setup encoder
    encoder = SentenceTransformerEncoder("all-MiniLM-L6-v2")
    encoders = EncoderManager(encoder_func=encoder)
    
    # Create encoder models and rules dictionaries
    encoder_models = {"all-MiniLM-L6-v2": encoder}
    encoder_rules = {"recentmem": "all-MiniLM-L6-v2", "longmem": "all-MiniLM-L6-v2"}
    
    # Set encoder attributes for compatibility
    encoders.models = encoder_models
    encoders.rules = encoder_rules
    
    # Memory prompt templates
    memory_prompts = {
        "workmem_to_recentmem": [
            "The following are some memories from {name}:",
            "{workmem}",
            "Please summarize the above memories into one concise, informative passage that preserves the key information."
        ],
        "recentmem_to_longmem": [
            "The following are some memories from {name}:",
            "{recentmem}",
            "Please summarize the above memories into one concise, informative passage that preserves the key information."
        ]
    }
    
    # Initialize LLM for summarization - using ChatOpenAI with o3-mini
    llm = ChatOpenAI(model_name="o3-mini")
    
    # Initialize memory modules with small capacities for demonstration
    memory_modules = {
        "workmem": MemoryStore(capacity=5),
        "recentmem": EmbeddingMemory(capacity=10, num_memories_queried=3, encoder=encoder, name="Story Memory Example"),
        "longmem": EmbeddingMemory(capacity=100, num_memories_queried=3, encoder=encoder, name="Story Memory Example")
    }
    
    # Create memory manager
    memory = ChunkedMemory(
        llm=llm,
        memory_modules=memory_modules,
        memory_prompts=memory_prompts,
        encoders=encoders,
        memory_bank=None,
        name="Story Memory Example"
    )
    
    # Sample story sentences from Little Prince
    story_sentences = [
        "Once upon a time, there was a little prince who lived on a very small planet.",
        "On his planet, there were three tiny volcanoes and a beautiful flower.",
        "The little prince loved to watch the sunsets on his small planet.",
        "He could see as many as forty-four sunsets in one day by moving his chair.",
        "The little prince decided to leave his planet to explore the universe.",
        "He visited many planets and met strange grown-ups.",
        "On one planet, he met a king who claimed to rule over everything.",
        "On another planet, he met a vain man who wanted to be admired by everyone.",
        "The third planet was inhabited by a drunkard who drank to forget.",
        "On Earth, the little prince met a fox who taught him about friendship.",
        "The fox said: 'One sees clearly only with the heart. What is essential is invisible to the eye.'",
        "The little prince realized that his flower was unique because of the time he had spent caring for it.",
        "After his journey, the little prince decided to return to his own planet."
    ]
    
    # Process each sentence from the story
    print("📚 Processing story sentences...\n")
    for i, sentence in enumerate(story_sentences):
        print(f"Input {i+1}/{len(story_sentences)}: {sentence}")
        memory.add(sentence)
        
        # Show memory transitions
        if memory.workmem_size_counter >= memory.workmem.capacity:
            print("\n⚙️ Working memory full - triggering summarization...")
            memory.update()
            print("  ↪ Memories moved to recent memory\n")
            
        if memory.recentmem_size_counter >= memory.recentmem.capacity:
            print("\n⚙️ Recent memory full - triggering summarization and clustering...")
            memory.update()
            print("  ↪ Memories summarized and moved to long-term memory\n")
    
    # Force final update to ensure all memories are processed
    memory.update()
    
    # Show memory contents
    print("\n📊 Final Memory State:\n")
    print("Working Memory:")
    for item in memory.workmem.items:
        print(f"  • {item}")
    
    print("\nRecent Memory:")
    for item in memory.recentmem.items:
        print(f"  • {item}")
    
    print("\nLong-Term Memory:")
    for item in memory.longmem.items:
        print(f"  • {item}")
    
    # Query example
    print("\n🔍 Memory Query Example:\n")
    query = "What did the fox teach the little prince?"
    print(f"Query: {query}")
    
    # Get memories related to the query
    memory_vars = memory.load_memory_variables({"text": query})
    
    print("\nRetrieved Memories:")
    if memory_vars["recentmem"]:
        print(f"From Recent Memory:\n  {memory_vars['recentmem']}")
    if memory_vars["longmem"]:
        print(f"From Long-Term Memory:\n  {memory_vars['longmem']}")

if __name__ == "__main__":
    main() 
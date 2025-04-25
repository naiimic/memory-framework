#!/usr/bin/env python3
"""
Test script to verify the memory system can be initialized correctly.
"""

import os
import sys
from dotenv import load_dotenv

# Add the parent directory to the path so we can import from the memory framework
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_architecture.module.default_modules import EmbeddingMemory, MemoryStore
from memory_architecture.manager.default_manager import ChunkedMemory
from utils.encoder_utils import EncoderManager, SentenceTransformerEncoder
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()

# Check if OpenAI API key is available
if not os.environ.get("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY not found in environment variables.")
    print("Please set your OpenAI API key in a .env file or as an environment variable.")
    sys.exit(1)

try:
    # Setup encoder
    encoder = SentenceTransformerEncoder("all-MiniLM-L6-v2")
    encoders = EncoderManager(encoder_func=encoder)
    
    # Create encoder models and rules dictionaries
    encoder_models = {"all-MiniLM-L6-v2": encoder}
    encoder_rules = {"shortmem": "all-MiniLM-L6-v2", "longmem": "all-MiniLM-L6-v2"}
    
    # Set encoder attributes for compatibility
    encoders.models = encoder_models
    encoders.rules = encoder_rules
    
    # Memory prompt templates
    memory_prompts = {
        "workmem_to_shortmem": [
            "The following are some memories from {name}:",
            "{workmem}",
            "Please summarize the above memories into one concise, informative passage that preserves the key information."
        ],
        "shortmem_to_longmem": [
            "The following are some memories from {name}:",
            "{shortmem}",
            "Please summarize the above memories into one concise, informative passage that preserves the key information."
        ]
    }
    
    # Initialize LLM for summarization
    llm = ChatOpenAI(temperature=0, model_name="o3-mini")
    
    # Initialize memory modules with small capacities for demonstration
    memory_modules = {
        "workmem": MemoryStore(capacity=3),
        "shortmem": EmbeddingMemory(capacity=5, num_memories_queried=3, encoder=encoder, name="Test Setup"),
        "longmem": EmbeddingMemory(capacity=100, num_memories_queried=3, encoder=encoder, name="Test Setup")
    }
    
    # Create memory manager
    memory = ChunkedMemory(
        llm=llm,
        memory_modules=memory_modules,
        memory_prompts=memory_prompts,
        encoders=encoders,
        memory_bank=None,
        name="Test Setup"
    )
    
    # Add a test memory
    memory.add("This is a test memory")
    
    print("✅ Memory system initialized successfully!")
    print("✅ All dependencies are properly installed.")
    print("✅ The examples should work correctly.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("The memory system could not be initialized correctly.")
    print("Please check your installation and dependencies.")
    sys.exit(1) 
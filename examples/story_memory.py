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
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import re
import random
from sklearn.cluster import DBSCAN

# Add the parent directory to the path so we can import the memory_architecture module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_architecture.module.default_modules import EmbeddingMemory, MemoryStore
from memory_architecture.manager.default_manager import ChunkedMemory
from utils.encoder_utils import EncoderManager, SentenceTransformerEncoder
from langchain_openai import ChatOpenAI
from sklearn.metrics.pairwise import cosine_similarity

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

# Create a tracked version of EmbeddingMemory to log forgetting events
class TrackedEmbeddingMemory(EmbeddingMemory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forgotten_count = 0
        
    def _update_memory_bank(self, new_embedding, new_memory_content):
        """Override to track forgetting events"""
        similarities = cosine_similarity(new_embedding.reshape(1, -1), self.embeddings)

        # Find indices of similar memories (similarities above or equal to the threshold)
        similar_indices = [
            i
            for i, value in enumerate(similarities[0])
            if value >= self.forgetting_threshold
        ]
        
        if similar_indices:
            self.forgotten_count += len(similar_indices)
            print(f"\n🗑️ FORGETTING TRIGGERED in {self.memory_type} memory!")
            print(f"  Number of similar memories being forgotten: {len(similar_indices)}")
            print(f"  Total forgotten so far: {self.forgotten_count}")
            print(f"  Forgetting threshold: {self.forgetting_threshold}")
            
            # Show the similar memories that will be forgotten
            for idx in similar_indices[:3]:  # Show at most 3 to avoid clutter
                print(f"  Forgotten memory: \"{self.memory_bank[idx]}\"")
                print(f"  Similarity score: {similarities[0][idx]:.4f}")
            
            if len(similar_indices) > 3:
                print(f"  ... and {len(similar_indices) - 3} more.")

        # Remove the similar memories and embeddings based on their indices
        for index in sorted(similar_indices, reverse=True):
            del self.memory_bank[index]
            del self.embeddings[index]

# Create a tracked version of ChunkedMemory that logs summarization
class TrackedMemory(ChunkedMemory):
    def summarize(self, memory_from: str = "shortmem", memory_to: str = "longmem"):
        print("\n🔍 CLUSTERING AND SUMMARIZATION INPUT:")
        print(f"  Short-term memories to be processed ({len(self.shortmem.items)} items):")
        for i, item in enumerate(self.shortmem.items):
            print(f"  {i+1}. {item}")
        
        dbscan = DBSCAN(eps=0.55, min_samples=1)
        labels = dbscan.fit_predict(getattr(self, memory_from).items_embeddings)
        
        results = super().summarize(memory_from, memory_to)
        
        print("\n🔄 CLUSTERING AND SUMMARIZATION RESULTS:")
        for i, summary in enumerate(results):
            print(f"  Cluster/Summary {i+1}: {summary}")
        print()
        
        return results

def main():
    print("\n🧠 Initializing Memory System...\n")
    
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
        "shortmem_to_longmem": [
            "The following are some memories from {name}:",
            "{shortmem}",
            "Please summarize the above memories into one concise, informative passage that preserves the key information."
        ]
    }
    
    # Initialize LLM for summarization - using ChatOpenAI with o3-mini
    llm = ChatOpenAI(model_name="o3-mini")
    
    # Initialize memory modules with small capacities for demonstration
    memory_modules = {
        "workmem": MemoryStore(capacity=5),
        "shortmem": TrackedEmbeddingMemory(capacity=50, num_memories_queried=3, 
                                            forgetting_threshold=0.9, memory_type="short-term",
                                            encoder=encoder, name="Story Memory Example"),
        "longmem": TrackedEmbeddingMemory(capacity=10000, num_memories_queried=3, 
                                           forgetting_threshold=0.9, memory_type="long-term",
                                           encoder=encoder, name="Story Memory Example")
    }
    
    # Create memory manager with our tracked version
    memory = TrackedMemory(
        llm=llm,
        memory_modules=memory_modules,
        memory_prompts=memory_prompts,
        encoders=encoders,
        memory_bank=None,
        name="Story Memory Example"
    )
    
    # Load text from examples/data/little_prince.txt
    story_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "little_prince.txt")
    
    try:
        with open(story_file_path, 'r', encoding='utf-8') as file:
            story_text = file.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
    
    # Split the text into sentences
    # This is a simple split by period, exclamation mark, and question mark
    # A more sophisticated sentence tokenizer would be better in practice
    
    # Clean up the text and split into sentences
    story_text = story_text.replace('\n', ' ').strip()
    sentences = re.split(r'(?<=[.!?])\s+', story_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Add some duplicate sentences to better demonstrate forgetting mechanism
    duplicated_sentences = []
    for i, sentence in enumerate(sentences[:100]):  # Take first 100 sentences
        if i % 10 == 0:  # Every 10th sentence
            duplicated_sentences.append(sentence)  # Add it twice
    
    # Insert duplicated sentences randomly in the second half
    for sentence in duplicated_sentences:
        position = random.randint(len(sentences)//2, len(sentences)-1)
        sentences.insert(position, sentence + " (duplicated)")
    
    # Set up tracking for memory sizes over time
    time_steps = []
    workmem_sizes = []
    shortmem_sizes = []
    longmem_sizes = []
    forgotten_counts_short = []
    forgotten_counts_long = []
    
    # Process each sentence from the story
    print(f"📚 Processing story from {story_file_path}...\n")
    print(f"Total sentences to process: {len(sentences)}\n")
    
    for i, sentence in enumerate(sentences):
        if i % 10 == 0:  # Print progress every 10 sentences
            print(f"Processing sentence {i+1}/{len(sentences)}")
        
        memory.add(sentence)
        
        # Track memory sizes
        time_steps.append(i)
        workmem_sizes.append(len(memory.workmem.items))
        shortmem_sizes.append(len(memory.shortmem.items))
        longmem_sizes.append(len(memory.longmem.items))
        forgotten_counts_short.append(memory.shortmem.forgotten_count)
        forgotten_counts_long.append(memory.longmem.forgotten_count)
        
        # Show memory transitions
        if memory.workmem_size_counter >= memory.workmem.capacity:
            # print("\n⚙️ Working memory full - transferring oldest memory...")
            oldest_memory = memory.workmem.items[0]
            # print(f"  Transferring: \"{oldest_memory}\"")
            memory.update()
            # print("  ↪ Oldest memory moved to Short-Term memory\n")
            
        if memory.shortmem_size_counter >= memory.shortmem.capacity:
            # print("\n⚙️ Short-Term memory full - triggering summarization and clustering...")
            memory.update()
            # print("  ↪ Memories summarized and moved to long-term memory\n")
    
    # Force final update to ensure all memories are processed
    memory.update()
    
    # Show memory contents
    print("\n📊 Final Memory State:\n")
    print("Working Memory:")
    for item in memory.workmem.items:
        print(f"  • {item}")
    
    print("\nShort-Term Memory:")
    for item in memory.shortmem.items:
        print(f"  • {item}")
    
    print("\nLong-Term Memory (sample of first 10 items):")
    for i, item in enumerate(memory.longmem.items[:10]):
        print(f"  • {item}")
    print(f"  ... plus {len(memory.longmem.items) - 10} more memories.")
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot memory sizes
    ax1.plot(time_steps, workmem_sizes, label='Working Memory', color='blue')
    ax1.plot(time_steps, shortmem_sizes, label='Short-Term Memory', color='green')
    ax1.plot(time_steps, longmem_sizes, label='Long-Term Memory', color='red')
    ax1.set_xlabel('Sentences Processed')
    ax1.set_ylabel('Number of Memories')
    ax1.set_title('Memory Size Evolution During Processing')
    ax1.legend()
    ax1.grid(True)
    
    # Plot forgotten counts
    ax2.plot(time_steps, forgotten_counts_short, label='Short-Term Forgotten', color='orange')
    ax2.plot(time_steps, forgotten_counts_long, label='Long-Term Forgotten', color='purple')
    ax2.set_xlabel('Sentences Processed')
    ax2.set_ylabel('Number of Forgotten Memories')
    ax2.set_title('Forgetting Events Over Time')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Save the plot in the examples directory
    plot_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory_evolution.png")
    plt.savefig(plot_file)
    print(f"\n📈 Memory evolution plot saved to {plot_file}")
    
    # Display plot if running in an interactive environment
    try:
        plt.show()
    except:
        pass
    
    # Print forgetting statistics
    print("\n📊 Forgetting Statistics:")
    print(f"  Short-Term Memory: {memory.shortmem.forgotten_count} memories forgotten")
    print(f"  Long-Term Memory: {memory.longmem.forgotten_count} memories forgotten")
    
    # Query example
    print("\n🔍 Memory Query Example:\n")
    query = "What did the fox teach the little prince?"
    print(f"Query: {query}")
    
    # Get memories related to the query
    memory_vars = memory.load_memory_variables({"text": query})
    
    print("\nRetrieved Memories:")
    if memory_vars["shortmem"]:
        print(f"From Short-Term memory:\n  {memory_vars['shortmem']}")
    if memory_vars["longmem"]:
        print(f"From Long-Term Memory:\n  {memory_vars['longmem']}")

if __name__ == "__main__":
    main() 
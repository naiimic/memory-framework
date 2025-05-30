#!/usr/bin/env python3
"""
Interactive Memory Demo

This interactive example demonstrates the Summarize-and-Forget memory framework
allowing users to add memories, trigger summarization, and query the memory system.
"""

import os
import sys
import cmd
import json
import logging
from dotenv import load_dotenv
import yaml

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

class MemoryShell(cmd.Cmd):
    intro = """
🧠 Welcome to the Interactive Memory System Demo 🧠
=================================================
This demo lets you interact with the Summarize-and-Forget memory framework.
Type 'help' to see available commands.
"""
    prompt = 'memory> '
    
    def __init__(self):
        super().__init__()
        print("Initializing memory system...")
        
        # Load memory configuration from YAML
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory_config.yaml")
        try:
            with open(config_path, 'r') as config_file:
                memory_config = yaml.safe_load(config_file)
            
            # Get memory capacities from config
            working_memory_capacity = memory_config['memory_capacities']['working_memory']
            short_term_memory_capacity = memory_config['memory_capacities']['short_term_memory']
            long_term_memory_capacity = memory_config['memory_capacities']['long_term_memory']
            
            print(f"\n📝 Loaded memory configuration from {config_path}")
            print(f"  Working Memory Capacity: {working_memory_capacity}")
            print(f"  Short-Term Memory Capacity: {short_term_memory_capacity}")
            print(f"  Long-Term Memory Capacity: {long_term_memory_capacity}\n")
            
        except FileNotFoundError:
            print(f"Warning: Configuration file {config_path} not found. Using default values.")
            working_memory_capacity = 3
            short_term_memory_capacity = 5
            long_term_memory_capacity = 100
        except Exception as e:
            print(f"Error loading configuration: {e}. Using default values.")
            working_memory_capacity = 3
            short_term_memory_capacity = 5
            long_term_memory_capacity = 100
        
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
        
        # Initialize LLM for summarization
        llm = ChatOpenAI(model_name="o3-mini")
        
        # Initialize memory modules with capacities from config
        memory_modules = {
            "workmem": MemoryStore(capacity=working_memory_capacity),
            "shortmem": EmbeddingMemory(capacity=short_term_memory_capacity, num_memories_queried=3, encoder=encoder, name="Interactive Memory Demo"),
            "longmem": EmbeddingMemory(capacity=long_term_memory_capacity, num_memories_queried=3, encoder=encoder, name="Interactive Memory Demo")
        }
        
        # Create memory manager
        self.memory = ChunkedMemory(
            llm=llm,
            memory_modules=memory_modules,
            memory_prompts=memory_prompts,
            encoders=encoders,
            memory_bank=None,
            name="Interactive Memory Demo"
        )
        
        print("Memory system ready!\n")
    
    def do_add(self, arg):
        """Add a new memory: add <text>"""
        if not arg:
            print("Error: Please provide text to add as a memory.")
            return
        
        self.memory.add(arg)
        print(f"Memory added: '{arg}'")
        
        # Show memory transitions if needed
        if self.memory.workmem_size_counter >= self.memory.workmem.capacity:
            print("Working memory capacity reached - oldest memory will be transferred to short-term memory on next update")
            
        if self.memory.shortmem_size_counter >= self.memory.shortmem.capacity:
            print("Short-term memory capacity reached - memories will be summarized on next update")
    
    def do_update(self, arg):
        """Trigger memory update/summarization process"""
        print("Updating memory system...")
        old_workmem_count = len(self.memory.workmem.items)
        old_shortmem_count = len(self.memory.shortmem.items)
        old_longmem_count = len(self.memory.longmem.items)
        
        self.memory.update()
        
        new_workmem_count = len(self.memory.workmem.items)
        new_shortmem_count = len(self.memory.shortmem.items)
        new_longmem_count = len(self.memory.longmem.items)
        
        print(f"Working memory: {old_workmem_count} → {new_workmem_count} items")
        print(f"Short-term memory: {old_shortmem_count} → {new_shortmem_count} items")
        print(f"Long-term memory: {old_longmem_count} → {new_longmem_count} items")
    
    def do_query(self, arg):
        """Query the memory system: query <text>"""
        if not arg:
            print("Error: Please provide text to query.")
            return
        
        print(f"Querying: '{arg}'")
        memory_vars = self.memory.load_memory_variables({"text": arg})
        
        print("\nRetrieved Memories:")
        if memory_vars["workmem"]:
            print(f"\nFrom Working Memory:\n  {memory_vars['workmem']}")
        if memory_vars["shortmem"]:
            print(f"\nFrom Short-Term Memory:\n  {memory_vars['shortmem']}")
        if memory_vars["longmem"]:
            print(f"\nFrom Long-Term Memory:\n  {memory_vars['longmem']}")
        
        if not memory_vars["shortmem"] and not memory_vars["longmem"]:
            print("  No relevant memories found.")
    
    def do_status(self, arg):
        """Show current memory system status"""
        workmem_count = len(self.memory.workmem.items) 
        shortmem_count = len(self.memory.shortmem.items)
        longmem_count = len(self.memory.longmem.items)
        
        print("\n📊 Memory System Status:")
        print(f"Working memory: {workmem_count}/{self.memory.workmem.capacity} items")
        print(f"Short-term memory: {shortmem_count}/{self.memory.shortmem.capacity} items")
        print(f"Long-term memory: {longmem_count}/{self.memory.longmem.capacity} items")
        
        print("\nWorking Memory Contents:")
        for i, item in enumerate(self.memory.workmem.items):
            print(f"  {i+1}. {item}")
        
        print("\nShort-Term Memory Contents:")
        for i, item in enumerate(self.memory.shortmem.items):
            print(f"  {i+1}. {item}")
        
        if longmem_count > 0:
            print("\nLong-Term Memory Contents (first 5):")
            for i, item in enumerate(self.memory.longmem.items[:5]):
                print(f"  {i+1}. {item}")
            if longmem_count > 5:
                print(f"  ... and {longmem_count-5} more")
    
    def do_clear(self, arg):
        """Clear all memories from the system"""
        self.memory.clear()
        print("Memory system cleared.")
    
    def do_exit(self, arg):
        """Exit the program"""
        print("Goodbye!")
        return True
    
    def do_quit(self, arg):
        """Exit the program"""
        return self.do_exit(arg)
    
    def do_help(self, arg):
        """List available commands with "help" or detailed help with "help cmd"."""
        if arg:
            # Show help for specific command
            super().do_help(arg)
        else:
            print("\nAvailable commands:")
            print("  add <text>     - Add a new memory")
            print("  update         - Trigger memory update/summarization")
            print("  query <text>   - Query the memory system")
            print("  status         - Show current memory system status")
            print("  clear          - Clear all memories")
            print("  exit           - Exit the program")
            print("  help           - Show this help message")
            print("  help <command> - Show detailed help for a command")

if __name__ == "__main__":
    MemoryShell().cmdloop() 
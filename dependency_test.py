#!/usr/bin/env python3
"""
Simple test script to verify basic dependencies.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

try:
    print("Testing imports...")
    
    # Testing imports
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    from langchain_openai import ChatOpenAI
    from sklearn.cluster import DBSCAN
    from langchain.chains import LLMChain
    from langchain_core.prompts import PromptTemplate
    
    print("✅ Basic imports successful")
    
    # Testing OPENAI_API_KEY
    if os.environ.get("OPENAI_API_KEY"):
        print("✅ OPENAI_API_KEY is set")
    else:
        print("❌ OPENAI_API_KEY is not set")
    
    # Testing sentence transformer model
    print("Testing sentence transformer model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(["This is a test sentence."])
    print(f"✅ SentenceTransformer works. Embedding shape: {embeddings.shape}")
    
    # Testing OpenAI API
    print("Testing OpenAI API...")
    llm = ChatOpenAI(model_name="o3-mini")
    prompt = PromptTemplate.from_template("Tell me a very short joke about {topic}")
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run(topic="programming")
    print(f"✅ OpenAI API works. Result: {result}")
    
    print("\nAll dependency tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Some dependencies could not be initialized correctly.")
    print("Please check your installation and dependencies.")
    sys.exit(1) 
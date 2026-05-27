#!/usr/bin/env python3
"""
Simple test script to check if llm_client_selection imports correctly
"""

try:
    print("Testing imports...")
    
    from llm_client_selection import (
        LLMClientSelectionSystem, 
        create_llm_client_selector,
        ClientMetrics,
        SelectionContext,
        OllamaLLMClient
    )
    
    print("✅ All imports successful!")
    
    # Test creating a simple selector
    print("Testing selector creation...")
    selector = create_llm_client_selector(4, strategy="adaptive")
    print("✅ Selector created successfully!")
    
    # Test Ollama client
    print("Testing Ollama client...")
    ollama_client = OllamaLLMClient()
    print(f"✅ Ollama client created! Available: {ollama_client.is_available()}")
    
    print("\n🎉 All tests passed! The module is working correctly.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()


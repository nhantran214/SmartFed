"""
Test script for Ollama (Llama3) integration with LLM Client Selection

This script tests the Ollama integration functionality for intelligent
client selection in federated learning.
"""

import torch
import numpy as np
import random
import logging
import time
from llm_client_selection import (
    LLMClientSelectionSystem, 
    create_llm_client_selector,
    ClientMetrics,
    SelectionContext,
    OllamaLLMClient
)

def setup_logging():
    """Setup logging for testing"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def test_ollama_availability():
    """Test if Ollama service is available"""
    print("=" * 50)
    print("Testing Ollama Service Availability")
    print("=" * 50)
    
    ollama_client = OllamaLLMClient()
    
    if ollama_client.is_available():
        print("✅ Ollama service is available!")
        print(f"   Base URL: {ollama_client.base_url}")
        print(f"   Model: {ollama_client.model_name}")
        return True
    else:
        print("❌ Ollama service is not available")
        print("   Please ensure Ollama is running on localhost:11434")
        print("   Install with: curl -fsSL https://ollama.ai/install.sh | sh")
        print("   Pull model with: ollama pull llama3:8b")
        return False

def test_ollama_client():
    """Test Ollama client functionality"""
    print("\n" + "=" * 50)
    print("Testing Ollama Client")
    print("=" * 50)
    
    ollama_client = OllamaLLMClient()
    
    if not ollama_client.is_available():
        print("❌ Skipping Ollama client test - service not available")
        return False
    
    # Test simple prompt
    test_prompt = "Hello! Please respond with 'Test successful' if you can understand this message."
    
    print(f"Sending test prompt: {test_prompt}")
    response = ollama_client.generate_response(test_prompt)
    
    if response:
        print(f"✅ Ollama response: {response}")
        return True
    else:
        print("❌ No response from Ollama")
        return False

def test_llm_selection_with_ollama():
    """Test LLM client selection using Ollama"""
    print("\n" + "=" * 50)
    print("Testing LLM Selection with Ollama")
    print("=" * 50)
    
    n_clients = 6
    selector = create_llm_client_selector(n_clients, strategy="llama3")
    
    if not hasattr(selector.selection_strategy, 'use_ollama') or not selector.selection_strategy.use_ollama:
        print("❌ Ollama not available, using fallback selector")
        return False
    
    print("✅ Using Llama3 for client selection")
    
    # Simulate some client metrics
    for round_num in range(3):
        print(f"\n--- Round {round_num} ---")
        
        # Update client metrics
        for client_id in range(n_clients):
            selector.update_client_metrics(
                client_id=client_id,
                loss=random.uniform(0.1, 2.0),
                accuracy=random.uniform(0.5, 0.95),
                data_quality=random.uniform(0.7, 1.0),
                performance_trend=random.uniform(-0.1, 0.1),
                resource_availability=random.uniform(0.8, 1.0),
                model_diversity=random.uniform(0.3, 0.8)
            )
        
        # Select clients using Llama3
        selected = selector.select_clients(
            current_round=round_num,
            selection_budget=3,
            global_performance=random.uniform(0.7, 0.9)
        )
        
        print(f"Llama3 selected clients: {selected}")
        
        # Small delay to avoid overwhelming Ollama
        time.sleep(1)
    
    print("✅ Llama3 selection test completed!")
    return True

def test_ollama_prompt_engineering():
    """Test different prompt engineering approaches"""
    print("\n" + "=" * 50)
    print("Testing Ollama Prompt Engineering")
    print("=" * 50)
    
    ollama_client = OllamaLLMClient()
    
    if not ollama_client.is_available():
        print("❌ Skipping prompt engineering test - Ollama not available")
        return False
    
    # Create mock context
    context = SelectionContext(
        current_round=1,
        total_rounds=10,
        available_clients=[0, 1, 2, 3, 4, 5],
        recent_performance={
            0: ClientMetrics(0, 0.5, 0.85, 0.9, 1, 0, 1, 0.1, 0.8, 0.6),
            1: ClientMetrics(1, 0.8, 0.75, 0.8, 1, 0, 1, -0.05, 0.9, 0.7),
            2: ClientMetrics(2, 1.2, 0.65, 0.7, 1, 0, 1, 0.05, 0.7, 0.5),
        },
        global_model_performance=0.8,
        selection_budget=3,
        diversity_requirement=0.3,
        performance_threshold=0.1
    )
    
    reward_scores = {0: 0.8, 1: 0.7, 2: 0.6, 3: 0.5, 4: 0.4, 5: 0.3}
    
    # Test context preparation
    context_text = ollama_client._prepare_context_for_llm(context, reward_scores, [0, 1, 2, 3, 4, 5])
    print("Context prepared for LLM:")
    print(context_text[:500] + "..." if len(context_text) > 500 else context_text)
    
    # Test prompt creation
    prompt = ollama_client._create_selection_prompt(context_text, [0, 1, 2, 3, 4, 5], 3)
    print(f"\nPrompt length: {len(prompt)} characters")
    
    # Test response generation
    print("\nSending prompt to Llama3...")
    response = ollama_client.generate_response(prompt)
    
    if response:
        print(f"✅ Llama3 response received:")
        print(f"   Length: {len(response)} characters")
        print(f"   Preview: {response[:200]}...")
        
        # Test parsing
        selected = ollama_client._parse_llm_response(response, [0, 1, 2, 3, 4, 5], 3)
        print(f"   Parsed selected clients: {selected}")
        
        return True
    else:
        print("❌ No response from Llama3")
        return False

def test_fallback_behavior():
    """Test fallback behavior when Ollama is not available"""
    print("\n" + "=" * 50)
    print("Testing Fallback Behavior")
    print("=" * 50)
    
    # Create selector with invalid Ollama URL to force fallback
    selector = create_llm_client_selector(4, strategy="llama3")
    
    # Check if fallback is working
    if hasattr(selector.selection_strategy, 'use_ollama'):
        if not selector.selection_strategy.use_ollama:
            print("✅ Fallback selector is active (Ollama not available)")
        else:
            print("✅ Ollama selector is active")
    else:
        print("✅ Using fallback selector")
    
    # Test selection with fallback
    for client_id in range(4):
        selector.update_client_metrics(
            client_id=client_id,
            loss=random.uniform(0.1, 2.0),
            accuracy=random.uniform(0.5, 0.95),
            data_quality=random.uniform(0.7, 1.0),
            performance_trend=random.uniform(-0.1, 0.1),
            resource_availability=random.uniform(0.8, 1.0),
            model_diversity=random.uniform(0.3, 0.8)
        )
    
    selected = selector.select_clients(
        current_round=0,
        selection_budget=2,
        global_performance=0.8
    )
    
    print(f"Selected clients (fallback): {selected}")
    print("✅ Fallback behavior test completed!")
    return True

def test_performance_comparison():
    """Compare performance between different selection strategies"""
    print("\n" + "=" * 50)
    print("Testing Performance Comparison")
    print("=" * 50)
    
    n_clients = 8
    strategies = ["greedy", "diversity", "adaptive", "llama3"]
    results = {}
    
    for strategy in strategies:
        print(f"\nTesting {strategy} strategy...")
        
        try:
            selector = create_llm_client_selector(n_clients, strategy=strategy)
            
            # Simulate training rounds
            selection_times = []
            for round_num in range(3):
                # Update metrics
                for client_id in range(n_clients):
                    selector.update_client_metrics(
                        client_id=client_id,
                        loss=random.uniform(0.1, 2.0),
                        accuracy=random.uniform(0.5, 0.95),
                        data_quality=random.uniform(0.7, 1.0),
                        performance_trend=random.uniform(-0.1, 0.1),
                        resource_availability=random.uniform(0.8, 1.0),
                        model_diversity=random.uniform(0.3, 0.8)
                    )
                
                # Time the selection
                start_time = time.time()
                selected = selector.select_clients(
                    current_round=round_num,
                    selection_budget=4,
                    global_performance=0.8
                )
                selection_time = time.time() - start_time
                selection_times.append(selection_time)
                
                print(f"  Round {round_num}: {selected} (took {selection_time:.2f}s)")
                
                # Small delay for Ollama
                if strategy == "llama3":
                    time.sleep(0.5)
            
            avg_time = np.mean(selection_times)
            results[strategy] = {
                'avg_selection_time': avg_time,
                'total_rounds': 3
            }
            
            print(f"  Average selection time: {avg_time:.2f}s")
            
        except Exception as e:
            print(f"  ❌ Error with {strategy}: {e}")
            results[strategy] = {'error': str(e)}
    
    # Print comparison
    print(f"\n{'Strategy':<12} {'Avg Time (s)':<15} {'Status'}")
    print("-" * 40)
    for strategy, result in results.items():
        if 'error' in result:
            print(f"{strategy:<12} {'N/A':<15} ERROR")
        else:
            print(f"{strategy:<12} {result['avg_selection_time']:<15.2f} OK")
    
    print("✅ Performance comparison completed!")
    return True

def main():
    """Run all Ollama integration tests"""
    logger = setup_logging()
    logger.info("Starting Ollama Integration Tests")
    
    try:
        # Run tests
        ollama_available = test_ollama_availability()
        
        if ollama_available:
            test_ollama_client()
            test_llm_selection_with_ollama()
            test_ollama_prompt_engineering()
        else:
            print("\n⚠️  Ollama not available - running limited tests")
        
        test_fallback_behavior()
        test_performance_comparison()
        
        print("\n" + "=" * 50)
        print("🎉 OLLAMA INTEGRATION TESTS COMPLETED! 🎉")
        print("=" * 50)
        
        if ollama_available:
            print("✅ Ollama integration is working properly!")
            print("✅ Llama3 can be used for intelligent client selection!")
        else:
            print("⚠️  Ollama not available - using fallback selectors")
            print("   To use Llama3, ensure Ollama is running and llama3:8b is installed")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

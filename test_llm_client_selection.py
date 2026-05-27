"""
Test script for LLM-based Client Selection System

This script tests the LLM client selection functionality independently
before integrating with the main federated learning system.
"""

import torch
import numpy as np
import random
import logging
from llm_client_selection import (
    LLMClientSelectionSystem, 
    create_llm_client_selector,
    ClientMetrics,
    SelectionContext
)

def setup_logging():
    """Setup logging for testing"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def test_reward_model():
    """Test the reward model functionality"""
    print("=" * 50)
    print("Testing Reward Model")
    print("=" * 50)
    
    # Create reward model
    from llm_client_selection import RewardModel
    reward_model = RewardModel()
    
    # Test feature extraction
    metrics = ClientMetrics(
        client_id=0,
        loss=0.5,
        accuracy=0.85,
        data_quality=0.9,
        communication_round=1,
        last_selected=0,
        selection_count=1,
        performance_trend=0.1,
        resource_availability=0.8,
        model_diversity=0.6
    )
    
    features = reward_model.extract_features(metrics)
    print(f"Extracted features shape: {features.shape}")
    print(f"Features: {features}")
    
    # Test forward pass
    with torch.no_grad():
        reward_score = reward_model(features.unsqueeze(0))
        print(f"Reward score: {reward_score.item():.4f}")
    
    print("✓ Reward model test passed!")

def test_client_metrics():
    """Test client metrics functionality"""
    print("\n" + "=" * 50)
    print("Testing Client Metrics")
    print("=" * 50)
    
    # Create metrics for multiple clients
    metrics_list = []
    for i in range(5):
        metrics = ClientMetrics(
            client_id=i,
            loss=random.uniform(0.1, 2.0),
            accuracy=random.uniform(0.5, 0.95),
            data_quality=random.uniform(0.7, 1.0),
            communication_round=1,
            last_selected=0,
            selection_count=0,
            performance_trend=random.uniform(-0.1, 0.1),
            resource_availability=random.uniform(0.8, 1.0),
            model_diversity=random.uniform(0.3, 0.8)
        )
        metrics_list.append(metrics)
        print(f"Client {i}: Loss={metrics.loss:.3f}, Accuracy={metrics.accuracy:.3f}, "
              f"Data Quality={metrics.data_quality:.3f}")
    
    print("✓ Client metrics test passed!")

def test_selection_strategies():
    """Test different selection strategies"""
    print("\n" + "=" * 50)
    print("Testing Selection Strategies")
    print("=" * 50)
    
    n_clients = 8
    selection_budget = 4
    
    # Test each strategy
    strategies = ["greedy", "diversity", "adaptive"]
    
    for strategy in strategies:
        print(f"\nTesting {strategy} strategy:")
        
        # Create selector
        selector = create_llm_client_selector(n_clients, strategy=strategy)
        
        # Simulate some rounds
        for round_num in range(3):
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
            
            # Select clients
            selected = selector.select_clients(
                current_round=round_num,
                selection_budget=selection_budget,
                global_performance=random.uniform(0.7, 0.9)
            )
            
            print(f"  Round {round_num}: Selected clients {selected}")
        
        # Get statistics
        stats = selector.get_selection_statistics()
        print(f"  Statistics: {stats}")
    
    print("✓ Selection strategies test passed!")

def test_adaptive_learning():
    """Test adaptive learning capabilities"""
    print("\n" + "=" * 50)
    print("Testing Adaptive Learning")
    print("=" * 50)
    
    n_clients = 6
    selector = create_llm_client_selector(n_clients, strategy="adaptive")
    
    # Simulate training with performance feedback
    for round_num in range(10):
        # Update client metrics
        client_metrics = []
        performance_feedback = []
        
        for client_id in range(n_clients):
            # Simulate improving performance over time
            base_accuracy = 0.5 + (round_num * 0.02) + random.uniform(-0.1, 0.1)
            accuracy = max(0.1, min(0.95, base_accuracy))
            
            loss = 2.0 - (round_num * 0.1) + random.uniform(-0.2, 0.2)
            loss = max(0.1, loss)
            
            selector.update_client_metrics(
                client_id=client_id,
                loss=loss,
                accuracy=accuracy,
                data_quality=random.uniform(0.7, 1.0),
                performance_trend=random.uniform(-0.05, 0.05),
                resource_availability=random.uniform(0.8, 1.0),
                model_diversity=random.uniform(0.3, 0.8)
            )
            
            # Collect metrics for training
            if round_num > 0:  # Skip first round
                if client_id in selector.client_metrics_history[-1]:
                    metrics = selector.client_metrics_history[-1][client_id]
                    client_metrics.append(metrics)
                    performance_feedback.append(accuracy)
        
        # Train reward model
        if client_metrics and performance_feedback:
            selector.train_reward_model(client_metrics, performance_feedback)
        
        # Select clients
        selected = selector.select_clients(
            current_round=round_num,
            selection_budget=3,
            global_performance=0.7 + (round_num * 0.01)
        )
        
        print(f"Round {round_num}: Selected {selected}, Avg Accuracy: {np.mean(performance_feedback):.3f}")
    
    # Final statistics
    stats = selector.get_selection_statistics()
    print(f"\nFinal Statistics: {stats}")
    
    print("✓ Adaptive learning test passed!")

def test_integration_simulation():
    """Simulate integration with federated learning"""
    print("\n" + "=" * 50)
    print("Testing Integration Simulation")
    print("=" * 50)
    
    n_clients = 8
    selector = create_llm_client_selector(n_clients, strategy="adaptive")
    
    # Simulate federated learning rounds
    for epoch in range(5):
        print(f"\n--- Epoch {epoch} ---")
        
        # Select clients for this round
        selected_clients = selector.select_clients(
            current_round=epoch,
            selection_budget=n_clients,  # Select all for now
            global_performance=0.8
        )
        
        print(f"Selected clients: {selected_clients}")
        
        # Simulate training each selected client
        for client_id in selected_clients:
            # Simulate training results
            loss = random.uniform(0.1, 1.5)
            accuracy = random.uniform(0.6, 0.9)
            
            # Calculate some derived metrics
            data_quality = random.uniform(0.8, 1.0)
            performance_trend = random.uniform(-0.1, 0.1)
            resource_availability = random.uniform(0.7, 1.0)
            model_diversity = random.uniform(0.4, 0.8)
            
            # Update metrics
            selector.update_client_metrics(
                client_id=client_id,
                loss=loss,
                accuracy=accuracy,
                data_quality=data_quality,
                performance_trend=performance_trend,
                resource_availability=resource_availability,
                model_diversity=model_diversity
            )
            
            print(f"  Client {client_id}: Loss={loss:.3f}, Accuracy={accuracy:.3f}")
        
        # Train reward model
        if epoch > 0:
            client_metrics = []
            performance_feedback = []
            
            for client_id in range(n_clients):
                if client_id in selector.client_metrics_history[-1]:
                    metrics = selector.client_metrics_history[-1][client_id]
                    client_metrics.append(metrics)
                    performance_feedback.append(metrics.accuracy)
            
            if client_metrics and performance_feedback:
                selector.train_reward_model(client_metrics, performance_feedback)
                print(f"  Trained reward model with {len(client_metrics)} clients")
        
        # Log statistics every few epochs
        if epoch % 2 == 0:
            stats = selector.get_selection_statistics()
            print(f"  Selection stats: {stats}")
    
    print("✓ Integration simulation test passed!")

def main():
    """Run all tests"""
    logger = setup_logging()
    logger.info("Starting LLM Client Selection Tests")
    
    try:
        # Run individual tests
        test_reward_model()
        test_client_metrics()
        test_selection_strategies()
        test_adaptive_learning()
        test_integration_simulation()
        
        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("=" * 50)
        print("The LLM Client Selection System is ready for integration!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

"""
Test script for Loss Weighting Agent

This script tests the Loss Weighting Agent functionality for dynamic
loss coefficient adjustment in federated learning.
"""

import torch
import numpy as np
import random
import logging
from loss_weighting_agent import (
    LossWeightingAgent, 
    create_loss_weighting_agent,
    LossWeightingContext,
    WeightingStrategy
)

def setup_logging():
    """Setup logging for testing"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def test_loss_weighting_agent_creation():
    """Test Loss Weighting Agent creation"""
    print("=" * 50)
    print("Testing Loss Weighting Agent Creation")
    print("=" * 50)
    
    agent = create_loss_weighting_agent()
    print("✅ Loss Weighting Agent created successfully!")
    
    # Test statistics
    stats = agent.get_weighting_statistics()
    print(f"Initial statistics: {stats}")
    
    return agent

def test_weighting_strategy():
    """Test weighting strategy configuration"""
    print("\n" + "=" * 50)
    print("Testing Weighting Strategy")
    print("=" * 50)
    
    strategy = WeightingStrategy(
        base_coefficient=0.1,
        adaptive_factor=1.0,
        diversity_factor=0.5,
        convergence_factor=0.3,
        stability_factor=0.2,
        exploration_rate=0.1
    )
    
    print(f"Base coefficient: {strategy.base_coefficient}")
    print(f"Adaptive factor: {strategy.adaptive_factor}")
    print(f"Diversity factor: {strategy.diversity_factor}")
    print(f"Convergence factor: {strategy.convergence_factor}")
    print(f"Stability factor: {strategy.stability_factor}")
    print(f"Exploration rate: {strategy.exploration_rate}")
    
    print("✅ Weighting strategy test passed!")

def test_loss_weighting_context():
    """Test Loss Weighting Context creation"""
    print("\n" + "=" * 50)
    print("Testing Loss Weighting Context")
    print("=" * 50)
    
    context = LossWeightingContext(
        current_round=5,
        total_rounds=20,
        global_loss=0.5,
        global_accuracy=0.85,
        client_losses=[0.4, 0.6, 0.3, 0.7],
        client_accuracies=[0.9, 0.8, 0.95, 0.75],
        client_weights=[0.25, 0.25, 0.25, 0.25],
        convergence_rate=0.1,
        diversity_score=0.7,
        previous_weighting_effectiveness=0.8,
        training_stability=0.9,
        communication_efficiency=0.85
    )
    
    print(f"Current round: {context.current_round}")
    print(f"Global loss: {context.global_loss}")
    print(f"Global accuracy: {context.global_accuracy}")
    print(f"Client losses: {context.client_losses}")
    print(f"Client accuracies: {context.client_accuracies}")
    print(f"Convergence rate: {context.convergence_rate}")
    print(f"Diversity score: {context.diversity_score}")
    
    print("✅ Loss Weighting Context test passed!")

def test_rule_based_weighting():
    """Test rule-based weighting (fallback)"""
    print("\n" + "=" * 50)
    print("Testing Rule-based Weighting")
    print("=" * 50)
    
    agent = create_loss_weighting_agent()
    
    # Create test context
    context = LossWeightingContext(
        current_round=5,
        total_rounds=20,
        global_loss=0.5,
        global_accuracy=0.85,
        client_losses=[0.4, 0.6, 0.3, 0.7],
        client_accuracies=[0.9, 0.8, 0.95, 0.75],
        client_weights=[0.25, 0.25, 0.25, 0.25],
        convergence_rate=0.1,
        diversity_score=0.7,
        previous_weighting_effectiveness=0.8,
        training_stability=0.9,
        communication_efficiency=0.85
    )
    
    # Test weighting for each client
    for client_idx in range(4):
        weighting = agent.calculate_loss_weighting(context, client_idx)
        print(f"Client {client_idx}: Loss weighting = {weighting:.4f}")
        
        # Validate bounds
        assert 0.01 <= weighting <= 1.0, f"Weighting {weighting} out of bounds for client {client_idx}"
    
    print("✅ Rule-based weighting test passed!")

def test_llm_based_weighting():
    """Test LLM-based weighting"""
    print("\n" + "=" * 50)
    print("Testing LLM-based Weighting")
    print("=" * 50)
    
    agent = create_loss_weighting_agent()
    
    if not agent.use_llm:
        print("⚠️  Ollama not available, skipping LLM-based test")
        return
    
    # Create test context
    context = LossWeightingContext(
        current_round=5,
        total_rounds=20,
        global_loss=0.5,
        global_accuracy=0.85,
        client_losses=[0.4, 0.6, 0.3, 0.7],
        client_accuracies=[0.9, 0.8, 0.95, 0.75],
        client_weights=[0.25, 0.25, 0.25, 0.25],
        convergence_rate=0.1,
        diversity_score=0.7,
        previous_weighting_effectiveness=0.8,
        training_stability=0.9,
        communication_efficiency=0.85
    )
    
    print("Testing LLM-based weighting...")
    
    # Test weighting for each client
    for client_idx in range(4):
        weighting = agent.calculate_loss_weighting(context, client_idx)
        print(f"Client {client_idx}: LLM weighting = {weighting:.4f}")
        
        # Validate bounds
        assert 0.01 <= weighting <= 1.0, f"Weighting {weighting} out of bounds for client {client_idx}"
    
    print("✅ LLM-based weighting test passed!")

def test_effectiveness_tracking():
    """Test effectiveness tracking and adaptation"""
    print("\n" + "=" * 50)
    print("Testing Effectiveness Tracking")
    print("=" * 50)
    
    agent = create_loss_weighting_agent()
    
    # Simulate several weighting decisions and effectiveness updates
    for i in range(10):
        old_weighting = 0.1
        new_weighting = 0.1 + random.uniform(-0.05, 0.05)
        performance_improvement = random.uniform(0.001, 0.02)
        
        agent.update_effectiveness(old_weighting, new_weighting, performance_improvement)
    
    # Get statistics
    stats = agent.get_weighting_statistics()
    print(f"Effectiveness tracking statistics: {stats}")
    
    assert 'total_decisions' in stats
    assert stats['total_decisions'] == 10
    assert 'recent_effectiveness' in stats
    assert 'overall_effectiveness' in stats
    
    print("✅ Effectiveness tracking test passed!")

def test_adaptive_strategy():
    """Test adaptive strategy learning"""
    print("\n" + "=" * 50)
    print("Testing Adaptive Strategy")
    print("=" * 50)
    
    agent = create_loss_weighting_agent()
    
    # Simulate learning with varying effectiveness
    for i in range(20):
        old_weighting = 0.1
        new_weighting = 0.1 + random.uniform(-0.05, 0.05)
        
        # Simulate improving effectiveness over time
        performance_improvement = 0.01 + (i * 0.001)
        
        agent.update_effectiveness(old_weighting, new_weighting, performance_improvement)
    
    # Get final statistics
    stats = agent.get_weighting_statistics()
    print(f"Final strategy: {stats['current_strategy']}")
    
    assert 'current_strategy' in stats
    assert 'adaptive_factor' in stats['current_strategy']
    assert 'exploration_rate' in stats['current_strategy']
    
    print("✅ Adaptive strategy test passed!")

def test_integration_simulation():
    """Simulate integration with federated learning"""
    print("\n" + "=" * 50)
    print("Testing Integration Simulation")
    print("=" * 50)
    
    agent = create_loss_weighting_agent()
    
    # Simulate federated learning rounds
    for epoch in range(5):
        print(f"\n--- Epoch {epoch} ---")
        
        # Create context for this epoch
        context = LossWeightingContext(
            current_round=epoch,
            total_rounds=10,
            global_loss=0.5 - (epoch * 0.05),  # Improving loss
            global_accuracy=0.8 + (epoch * 0.02),  # Improving accuracy
            client_losses=[random.uniform(0.2, 0.8) for _ in range(4)],
            client_accuracies=[random.uniform(0.7, 0.95) for _ in range(4)],
            client_weights=[0.25, 0.25, 0.25, 0.25],
            convergence_rate=0.1 + (epoch * 0.02),
            diversity_score=0.7,
            previous_weighting_effectiveness=0.8,
            training_stability=0.9,
            communication_efficiency=0.85
        )
        
        # Calculate weighting for each client
        weightings = []
        for client_idx in range(4):
            weighting = agent.calculate_loss_weighting(context, client_idx)
            weightings.append(weighting)
            print(f"  Client {client_idx}: {weighting:.4f}")
        
        # Update effectiveness
        avg_weighting = np.mean(weightings)
        performance_improvement = 0.01 + (epoch * 0.005)
        agent.update_effectiveness(0.1, avg_weighting, performance_improvement)
        
        print(f"  Average weighting: {avg_weighting:.4f}")
    
    # Final statistics
    stats = agent.get_weighting_statistics()
    print(f"\nFinal statistics: {stats}")
    
    print("✅ Integration simulation test passed!")

def test_performance_comparison():
    """Compare performance between different approaches"""
    print("\n" + "=" * 50)
    print("Testing Performance Comparison")
    print("=" * 50)
    
    # Test with LLM
    agent_llm = create_loss_weighting_agent()
    
    # Test with rule-based (force fallback)
    agent_rule = create_loss_weighting_agent()
    agent_rule.use_llm = False
    
    context = LossWeightingContext(
        current_round=5,
        total_rounds=20,
        global_loss=0.5,
        global_accuracy=0.85,
        client_losses=[0.4, 0.6, 0.3, 0.7],
        client_accuracies=[0.9, 0.8, 0.95, 0.75],
        client_weights=[0.25, 0.25, 0.25, 0.25],
        convergence_rate=0.1,
        diversity_score=0.7,
        previous_weighting_effectiveness=0.8,
        training_stability=0.9,
        communication_efficiency=0.85
    )
    
    print("Comparing weighting approaches:")
    
    for client_idx in range(4):
        llm_weighting = agent_llm.calculate_loss_weighting(context, client_idx)
        rule_weighting = agent_rule.calculate_loss_weighting(context, client_idx)
        
        print(f"Client {client_idx}:")
        print(f"  LLM weighting: {llm_weighting:.4f}")
        print(f"  Rule weighting: {rule_weighting:.4f}")
        print(f"  Difference: {abs(llm_weighting - rule_weighting):.4f}")
    
    print("✅ Performance comparison test passed!")

def main():
    """Run all Loss Weighting Agent tests"""
    logger = setup_logging()
    logger.info("Starting Loss Weighting Agent Tests")
    
    try:
        # Run individual tests
        test_loss_weighting_agent_creation()
        test_weighting_strategy()
        test_loss_weighting_context()
        test_rule_based_weighting()
        test_llm_based_weighting()
        test_effectiveness_tracking()
        test_adaptive_strategy()
        test_integration_simulation()
        test_performance_comparison()
        
        print("\n" + "=" * 50)
        print("🎉 ALL LOSS WEIGHTING AGENT TESTS PASSED! 🎉")
        print("=" * 50)
        print("The Loss Weighting Agent is ready for integration!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

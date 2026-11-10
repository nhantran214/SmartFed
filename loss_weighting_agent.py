"""
Loss Weighting Agent using LLM for Dynamic Loss Coefficient Adjustment

This module implements an intelligent agent that uses LLM to dynamically
adjust loss weighting coefficients in federated learning based on:
1. Current training performance
2. Client diversity and quality
3. Convergence patterns
4. Historical effectiveness

Author: Nhan Tran
"""

import torch
import torch.nn as nn
import numpy as np
import json
import logging
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
import random
import requests
import time
from llm_client_selection import OllamaLLMClient

@dataclass
class LossWeightingContext:
    """Context for loss weighting decision"""
    current_round: int
    total_rounds: int
    global_loss: float
    global_accuracy: float
    client_losses: List[float]
    client_accuracies: List[float]
    client_weights: List[float]
    convergence_rate: float
    diversity_score: float
    previous_weighting_effectiveness: float
    training_stability: float
    communication_efficiency: float

@dataclass
class WeightingStrategy:
    """Loss weighting strategy configuration"""
    base_coefficient: float
    adaptive_factor: float
    diversity_factor: float
    convergence_factor: float
    stability_factor: float
    exploration_rate: float

class LossWeightingAgent:
    """
    Intelligent agent for dynamic loss weighting using LLM
    """
    
    def __init__(self, 
                 model_name: str = "llama3:8b",
                 base_url: str = "http://localhost:11434",
                 logger: Optional[logging.Logger] = None,
                 use_llm: Optional[bool] = None):
        """
        Initialize Loss Weighting Agent
        
        Args:
            model_name: LLM model name for Ollama
            base_url: Ollama API base URL
            logger: Logger instance
            use_llm: Force LLM on/off. False = pure Eq.(26) rule-based (no LLM calls).
                     None = auto from Ollama availability.
        """
        self.ollama_client = OllamaLLMClient(model_name, base_url)
        self.logger = logger or logging.getLogger(__name__)
        
        # Weighting history for learning
        self.weighting_history = []
        self.effectiveness_history = []
        self.performance_history = []
        self.fallback_count = 0
        
        # Default strategy
        self.current_strategy = WeightingStrategy(
            base_coefficient=0.1,
            adaptive_factor=1.0,
            diversity_factor=0.5,
            convergence_factor=0.3,
            stability_factor=0.2,
            exploration_rate=0.1
        )
        
        # Learning parameters
        self.learning_rate = 0.01
        self.memory_size = 50
        self.adaptation_threshold = 0.05
        
        # Resolve use_llm: explicit False never calls LLM (R3-Major-4 rule-based baseline).
        if use_llm is False:
            self.use_llm = False
            self.logger.info("Loss Weighting Agent: rule-based Eq.(26) only (use_llm=False)")
        elif use_llm is True:
            if not self.ollama_client.is_available():
                self.logger.warning("Ollama not available; AFWA requested LLM but will use rule-based")
                self.use_llm = False
            else:
                self.logger.info(f"Loss Weighting Agent initialized with {model_name}")
                self.use_llm = True
        else:
            if not self.ollama_client.is_available():
                self.logger.warning("Ollama not available, using rule-based weighting")
                self.use_llm = False
            else:
                self.logger.info(f"Loss Weighting Agent initialized with {model_name}")
                self.use_llm = True
    
    def calculate_loss_weighting(self, 
                               context: LossWeightingContext,
                               client_index: int) -> float:
        """
        Calculate optimal loss weighting coefficient for a specific client
        
        Args:
            context: Current training context
            client_index: Index of the client
            
        Returns:
            Optimal loss weighting coefficient
        """
        if self.use_llm:
            return self._clip_coefficient(self._llm_based_weighting(context, client_index))
        return self._clip_coefficient(self._rule_based_weighting(context, client_index))

    @staticmethod
    def _coeff_bounds():
        try:
            from config import Config as _Cfg
            lo = float(getattr(_Cfg, 'AFWA_COEFF_MIN', 0.01))
            hi = float(getattr(_Cfg, 'AFWA_COEFF_MAX', 1.0))
        except Exception:
            lo, hi = 0.01, 1.0
        if lo > hi:
            lo, hi = hi, lo
        return lo, hi

    def _clip_coefficient(self, coeff):
        lo, hi = self._coeff_bounds()
        return max(lo, min(hi, float(coeff)))
    
    def _llm_based_weighting(self, 
                           context: LossWeightingContext, 
                           client_index: int) -> float:
        """Use LLM to determine optimal loss weighting"""
        
        try:
            # Prepare context for LLM
            context_text = self._prepare_weighting_context(context, client_index)
            
            # Create prompt for LLM
            prompt = self._create_weighting_prompt(context_text, client_index)
            
            # Get LLM response
            response = self.ollama_client.generate_response(prompt)
            if not response:
                self.logger.warning(
                    f"Empty LLM response for client {client_index}; using rule-based weighting"
                )
                # Empty/timeout already counted in OllamaLLMClient.generate_response.
                self.fallback_count += 1
                return self._rule_based_weighting(context, client_index)
            
            # Parse response to get weighting coefficient
            weighting_coeff = self._parse_weighting_response(response, context, client_index)
            
            self.logger.info(f"LLM determined weighting coefficient: {weighting_coeff:.4f}")
            return weighting_coeff
            
        except Exception as e:
            self.logger.error(f"Error in LLM-based weighting: {e}")
            self._mark_fallback()
            return self._rule_based_weighting(context, client_index)

    def _mark_fallback(self):
        """Count parse/API fallbacks that activate Eq.(26)."""
        self.fallback_count += 1
        try:
            from llm_client_selection import GLOBAL_LLM_COST
            GLOBAL_LLM_COST.record_parse_fallback()
        except Exception:
            pass
    
    def _rule_based_weighting(self, 
                            context: LossWeightingContext, 
                            client_index: int) -> float:
        """Rule-based fallback for loss weighting"""
        
        # Base coefficient
        base_coeff = self.current_strategy.base_coefficient
        
        # Client-specific adjustments
        client_loss = context.client_losses[client_index]
        client_accuracy = context.client_accuracies[client_index]
        client_weight = context.client_weights[client_index]
        
        # Performance-based adjustment
        performance_factor = 1.0
        if client_accuracy > context.global_accuracy:
            performance_factor += 0.2  # Reward better performance
        if client_loss < context.global_loss:
            performance_factor += 0.1  # Reward lower loss
        
        # Diversity-based adjustment
        diversity_factor = 1.0 + (context.diversity_score * 0.3)
        
        # Convergence-based adjustment
        convergence_factor = 1.0
        if context.convergence_rate > 0.1:
            convergence_factor += 0.2  # Increase weighting for fast convergence
        elif context.convergence_rate < 0.01:
            convergence_factor -= 0.1  # Decrease weighting for slow convergence
        
        # Stability-based adjustment
        stability_factor = 1.0 + (context.training_stability * 0.2)
        
        # Calculate final coefficient
        final_coeff = (base_coeff * performance_factor * diversity_factor * 
                      convergence_factor * stability_factor)
        
        # Apply bounds
        final_coeff = self._clip_coefficient(final_coeff)
        
        return final_coeff
    
    def _prepare_weighting_context(self, 
                                 context: LossWeightingContext, 
                                 client_index: int) -> str:
        """Prepare context information for LLM"""
        
        context_info = f"""
LOSS WEIGHTING DECISION CONTEXT:

Training Round: {context.current_round}/{context.total_rounds}
Global Performance:
  - Global Loss: {context.global_loss:.4f}
  - Global Accuracy: {context.global_accuracy:.4f}
  - Convergence Rate: {context.convergence_rate:.4f}
  - Training Stability: {context.training_stability:.4f}
  - Diversity Score: {context.diversity_score:.4f}
  - Communication Efficiency: {context.communication_efficiency:.4f}

Client {client_index} Metrics:
  - Client Loss: {context.client_losses[client_index]:.4f}
  - Client Accuracy: {context.client_accuracies[client_index]:.4f}
  - Client Weight: {context.client_weights[client_index]:.4f}
  - Loss vs Global: {context.client_losses[client_index] - context.global_loss:+.4f}
  - Accuracy vs Global: {context.client_accuracies[client_index] - context.global_accuracy:+.4f}

All Clients Performance:
"""
        
        for i, (loss, acc, weight) in enumerate(zip(context.client_losses, 
                                                   context.client_accuracies, 
                                                   context.client_weights)):
            context_info += f"  Client {i}: Loss={loss:.4f}, Acc={acc:.4f}, Weight={weight:.4f}\n"
        
        # Add historical context
        if self.weighting_history:
            recent_effectiveness = np.mean(self.effectiveness_history[-5:]) if self.effectiveness_history else 0.0
            context_info += f"\nHistorical Context:\n"
            context_info += f"  - Recent Weighting Effectiveness: {recent_effectiveness:.4f}\n"
            context_info += f"  - Previous Weighting Effectiveness: {context.previous_weighting_effectiveness:.4f}\n"
        
        return context_info
    
    def _create_weighting_prompt(self, 
                               context_text: str, 
                               client_index: int) -> str:
        """Create prompt for LLM to determine loss weighting"""
        lo, hi = self._coeff_bounds()
        
        prompt = f"""
{context_text}

TASK: Determine the optimal loss weighting coefficient for Client {client_index} in federated learning.

CURRENT SITUATION:
- We need to weight the loss contribution of Client {client_index}
- Current base coefficient is 0.1
- The coefficient affects how much this client's loss contributes to the global model update

WEIGHTING CRITERIA:
1. Performance: Better performing clients should have higher influence
2. Diversity: Clients with unique data/characteristics should be weighted appropriately
3. Convergence: Fast-converging clients should be rewarded
4. Stability: Stable clients should be trusted more
5. Fairness: Avoid over-weighting any single client
6. Efficiency: Balance between performance and communication efficiency

CONSTRAINTS:
- Coefficient should be between {lo:.2f} and {hi:.2f}
- Consider the client's performance relative to global average
- Account for training stability and convergence patterns
- Maintain diversity in the federated learning process

RESPONSE FORMAT:
Optimal Coefficient: [value between {lo:.2f} and {hi:.2f}]
Reasoning: [Brief explanation of the decision]

Please determine the optimal loss weighting coefficient for Client {client_index}.
"""
        
        return prompt
    
    def _parse_weighting_response(self, 
                                response: str, 
                                context: LossWeightingContext,
                                client_index: int) -> float:
        """Parse LLM response to extract weighting coefficient"""
        
        try:
            # Look for "Optimal Coefficient:" pattern
            lines = response.split('\n')
            coefficient = None
            
            for line in lines:
                if "Optimal Coefficient:" in line or "coefficient:" in line.lower():
                    # Extract the number
                    import re
                    numbers = re.findall(r'(\d+\.?\d*)', line)
                    if numbers:
                        coefficient = float(numbers[0])
                        break
            
            # If no coefficient found, try to extract any number
            if coefficient is None:
                import re
                numbers = re.findall(r'(\d+\.?\d*)', response)
                if numbers:
                    coefficient = float(numbers[0])
            
            # Validate and apply bounds
            if coefficient is not None:
                coefficient = self._clip_coefficient(coefficient)
            else:
                # Fallback to rule-based
                self._mark_fallback()
                coefficient = self._rule_based_weighting(context, client_index)
            
            return coefficient
            
        except Exception as e:
            self.logger.warning(f"Error parsing weighting response: {e}")
            self._mark_fallback()
            return self._rule_based_weighting(context, client_index)
    
    def update_effectiveness(self, 
                           old_weighting: float,
                           new_weighting: float,
                           performance_improvement: float):
        """Update effectiveness tracking for learning"""
        
        # Calculate effectiveness
        effectiveness = performance_improvement / (abs(new_weighting - old_weighting) + 1e-8)
        
        # Store in history
        self.effectiveness_history.append(effectiveness)
        self.weighting_history.append({
            'old_weighting': old_weighting,
            'new_weighting': new_weighting,
            'effectiveness': effectiveness,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep memory size limited
        if len(self.effectiveness_history) > self.memory_size:
            self.effectiveness_history.pop(0)
            self.weighting_history.pop(0)
        
        # Adapt strategy based on effectiveness
        self._adapt_strategy()
    
    def _adapt_strategy(self):
        """Adapt weighting strategy based on historical effectiveness"""
        
        if len(self.effectiveness_history) < 5:
            return
        
        recent_effectiveness = np.mean(self.effectiveness_history[-5:])
        overall_effectiveness = np.mean(self.effectiveness_history)
        
        # If recent performance is significantly different, adapt
        if abs(recent_effectiveness - overall_effectiveness) > self.adaptation_threshold:
            if recent_effectiveness > overall_effectiveness:
                # Recent performance is better, increase exploration
                self.current_strategy.exploration_rate *= 1.1
                self.current_strategy.adaptive_factor *= 1.05
            else:
                # Recent performance is worse, decrease exploration
                self.current_strategy.exploration_rate *= 0.9
                self.current_strategy.adaptive_factor *= 0.95
            
            # Apply bounds
            self.current_strategy.exploration_rate = max(0.01, min(0.5, self.current_strategy.exploration_rate))
            self.current_strategy.adaptive_factor = max(0.1, min(2.0, self.current_strategy.adaptive_factor))
            
            self.logger.info(f"Adapted strategy: exploration={self.current_strategy.exploration_rate:.3f}, "
                           f"adaptive={self.current_strategy.adaptive_factor:.3f}")
    
    def get_weighting_statistics(self) -> Dict[str, Any]:
        """Get statistics about loss weighting performance"""
        
        if not self.weighting_history:
            return {}
        
        recent_effectiveness = np.mean(self.effectiveness_history[-10:]) if len(self.effectiveness_history) >= 10 else 0.0
        overall_effectiveness = np.mean(self.effectiveness_history)
        
        return {
            'total_decisions': len(self.weighting_history),
            'recent_effectiveness': recent_effectiveness,
            'overall_effectiveness': overall_effectiveness,
            'current_strategy': {
                'base_coefficient': self.current_strategy.base_coefficient,
                'adaptive_factor': self.current_strategy.adaptive_factor,
                'diversity_factor': self.current_strategy.diversity_factor,
                'convergence_factor': self.current_strategy.convergence_factor,
                'stability_factor': self.current_strategy.stability_factor,
                'exploration_rate': self.current_strategy.exploration_rate
            },
            'use_llm': self.use_llm
        }

def create_loss_weighting_agent(model_name: str = "llama3:8b",
                               logger: Optional[logging.Logger] = None,
                               use_llm: Optional[bool] = None) -> LossWeightingAgent:
    """
    Factory function to create Loss Weighting Agent
    
    Args:
        model_name: LLM model name
        logger: Logger instance
        use_llm: Force LLM on/off (False = pure rule-based Eq.26 baseline)
        
    Returns:
        LossWeightingAgent instance
    """
    return LossWeightingAgent(model_name=model_name, logger=logger, use_llm=use_llm)

# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create loss weighting agent
    agent = create_loss_weighting_agent(logger=logger)
    
    # Test with mock context
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
    
    # Test weighting calculation
    for client_idx in range(4):
        weighting = agent.calculate_loss_weighting(context, client_idx)
        print(f"Client {client_idx}: Loss weighting = {weighting:.4f}")
    
    # Test statistics
    stats = agent.get_weighting_statistics()
    print(f"Weighting statistics: {stats}")

"""
LLM-based Adaptive Client Selection for Federated Learning

This module implements an intelligent client selection system that uses:
1. Reward Model to evaluate client performance
2. LLM to make intelligent selection decisions
3. Adaptive strategies that evolve over time

Author: Nhan Tran
"""

import torch
import torch.nn as nn
import numpy as np
import json
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import random
from datetime import datetime
import os
import requests
import time

@dataclass
class ClientMetrics:
    """Metrics for a single client"""
    client_id: int
    loss: float
    accuracy: float
    data_quality: float
    communication_round: int
    last_selected: int
    selection_count: int
    performance_trend: float  # Trend over last N rounds
    resource_availability: float  # 0-1, higher is better
    model_diversity: float  # How different this client's model is from others

class OllamaLLMClient:
    """Client for communicating with Ollama API"""
    
    def __init__(self, model_name: str = "llama3:8b", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        self.logger = logging.getLogger(__name__)
    
    def is_available(self) -> bool:
        """Check if Ollama service is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def generate_response(self, prompt: str, max_retries: int = 3) -> str:
        """Generate response from Llama3 model"""
        for attempt in range(max_retries):
            try:
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 1000
                    }
                }
                
                response = requests.post(
                    self.api_url,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get("response", "").strip()
                else:
                    self.logger.warning(f"Ollama API error: {response.status_code}")
                    
            except requests.exceptions.Timeout:
                self.logger.warning(f"Ollama API timeout (attempt {attempt + 1})")
            except Exception as e:
                self.logger.warning(f"Ollama API error (attempt {attempt + 1}): {e}")
            
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return ""
    
    def select_clients_llm(self, 
                          context: 'SelectionContext', 
                          reward_scores: Dict[int, float],
                          available_clients: List[int]) -> List[int]:
        """Use Llama3 to make client selection decision"""
        
        # Prepare context for LLM
        context_text = self._prepare_context_for_llm(context, reward_scores, available_clients)
        
        # Create prompt for Llama3
        prompt = self._create_selection_prompt(context_text, available_clients, context.selection_budget)
        
        # Get LLM response
        response = self.generate_response(prompt)
        
        # Parse response to get selected clients
        selected_clients = self._parse_llm_response(response, available_clients, context.selection_budget)
        
        self.logger.info(f"Llama3 selection response: {response[:200]}...")
        self.logger.info(f"Parsed selected clients: {selected_clients}")
        
        return selected_clients
    
    def _prepare_context_for_llm(self, 
                                context: 'SelectionContext', 
                                reward_scores: Dict[int, float],
                                available_clients: List[int]) -> str:
        """Prepare context information for LLM"""
        
        context_info = f"""
FEDERATED LEARNING CLIENT SELECTION CONTEXT:

Current Round: {context.current_round}
Total Rounds: {context.total_rounds}
Selection Budget: {context.selection_budget} clients needed
Global Model Performance: {context.global_model_performance:.3f}
Diversity Requirement: {context.diversity_requirement:.3f}
Performance Threshold: {context.performance_threshold:.3f}

AVAILABLE CLIENTS AND THEIR METRICS:
"""
        
        for client_id in available_clients:
            if client_id in context.recent_performance:
                metrics = context.recent_performance[client_id]
                reward_score = reward_scores.get(client_id, 0.0)
                
                context_info += f"""
Client {client_id}:
  - Loss: {metrics.loss:.4f}
  - Accuracy: {metrics.accuracy:.4f}
  - Data Quality: {metrics.data_quality:.4f}
  - Performance Trend: {metrics.performance_trend:.4f}
  - Resource Availability: {metrics.resource_availability:.4f}
  - Model Diversity: {metrics.model_diversity:.4f}
  - Selection Count: {metrics.selection_count}
  - Last Selected: {context.current_round - metrics.last_selected} rounds ago
  - Reward Score: {reward_score:.4f}
"""
            else:
                reward_score = reward_scores.get(client_id, 0.0)
                context_info += f"""
Client {client_id}:
  - No recent performance data
  - Reward Score: {reward_score:.4f}
"""
        
        return context_info
    
    def _create_selection_prompt(self, 
                               context_text: str, 
                               available_clients: List[int],
                               selection_budget: int) -> str:
        """Create prompt for Llama3 to make selection decision"""
        
        prompt = f"""
{context_text}

TASK: Select {selection_budget} clients for federated learning training.

SELECTION CRITERIA:
1. Performance: Higher accuracy and lower loss are better
2. Diversity: Prefer clients with different model characteristics
3. Resource Availability: Clients with better resource availability
4. Fairness: Consider clients that haven't been selected recently
5. Data Quality: Prefer clients with higher data quality
6. Trend: Consider performance improvement trends

INSTRUCTIONS:
- Analyze each client's metrics carefully
- Consider the balance between performance and diversity
- Ensure fair selection (don't always pick the same clients)
- Select exactly {selection_budget} clients
- Provide reasoning for your selection

RESPONSE FORMAT:
Selected Clients: [client_id1, client_id2, client_id3, ...]
Reasoning: [Brief explanation of selection rationale]

Please select the best {selection_budget} clients from the available clients: {available_clients}
"""
        
        return prompt
    
    def _parse_llm_response(self, 
                          response: str, 
                          available_clients: List[int],
                          selection_budget: int) -> List[int]:
        """Parse LLM response to extract selected clients"""
        
        try:
            # Look for "Selected Clients:" pattern
            lines = response.split('\n')
            selected_clients = []
            
            for line in lines:
                if "Selected Clients:" in line or "selected clients:" in line.lower():
                    # Extract the list part
                    if '[' in line and ']' in line:
                        list_part = line[line.find('['):line.find(']') + 1]
                        try:
                            # Parse the list
                            selected_clients = eval(list_part)
                            break
                        except:
                            continue
            
            # If no list found, try to extract numbers from the response
            if not selected_clients:
                import re
                numbers = re.findall(r'\b(\d+)\b', response)
                selected_clients = [int(n) for n in numbers if int(n) in available_clients]
            
            # Validate and limit to selection budget
            selected_clients = [c for c in selected_clients if c in available_clients]
            selected_clients = selected_clients[:selection_budget]
            
            # If still not enough, add random clients
            if len(selected_clients) < selection_budget:
                remaining = [c for c in available_clients if c not in selected_clients]
                needed = selection_budget - len(selected_clients)
                selected_clients.extend(random.sample(remaining, min(needed, len(remaining))))
            
            return selected_clients
            
        except Exception as e:
            self.logger.warning(f"Error parsing LLM response: {e}")
            # Fallback to random selection
            return random.sample(available_clients, min(selection_budget, len(available_clients)))

@dataclass
class SelectionContext:
    """Context for client selection decision"""
    current_round: int
    total_rounds: int
    available_clients: List[int]
    recent_performance: Dict[int, ClientMetrics]
    global_model_performance: float
    selection_budget: int  # How many clients to select
    diversity_requirement: float  # How much diversity is needed
    performance_threshold: float  # Minimum performance to consider

class RewardModel(nn.Module):
    """
    Neural network-based reward model that evaluates client performance
    based on multiple metrics and historical data
    """
    
    def __init__(self, input_dim: int = 10, hidden_dim: int = 64, output_dim: int = 1):
        super(RewardModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        # Reward prediction head
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid()  # Output between 0 and 1
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute reward score
        
        Args:
            features: Input features [batch_size, input_dim]
            
        Returns:
            reward_scores: Reward scores [batch_size, 1]
        """
        features = self.feature_extractor(features)
        reward_scores = self.reward_head(features)
        return reward_scores
    
    def extract_features(self, metrics: ClientMetrics) -> torch.Tensor:
        """
        Extract features from client metrics
        
        Args:
            metrics: ClientMetrics object
            
        Returns:
            features: Feature tensor [input_dim]
        """
        features = torch.tensor([
            metrics.loss,
            metrics.accuracy,
            metrics.data_quality,
            metrics.performance_trend,
            metrics.resource_availability,
            metrics.model_diversity,
            metrics.selection_count / (metrics.communication_round + 1),  # Selection frequency
            (metrics.communication_round - metrics.last_selected) / 10.0,  # Time since last selection
            min(metrics.performance_trend, 1.0),  # Capped trend
            random.random()  # Random factor for exploration
        ], dtype=torch.float32)
        
        return features

class LLMClientSelector(ABC):
    """Abstract base class for LLM-based client selection strategies"""
    
    @abstractmethod
    def select_clients(self, context: SelectionContext, reward_scores: Dict[int, float]) -> List[int]:
        """
        Select clients based on context and reward scores
        
        Args:
            context: Selection context
            reward_scores: Reward scores for each client
            
        Returns:
            selected_clients: List of selected client IDs
        """
        pass

class GreedyLLMSelector(LLMClientSelector):
    """Greedy selection based on reward scores"""
    
    def select_clients(self, context: SelectionContext, reward_scores: Dict[int, float]) -> List[int]:
        """Select top-k clients based on reward scores"""
        # Sort clients by reward score
        sorted_clients = sorted(
            context.available_clients,
            key=lambda x: reward_scores.get(x, 0.0),
            reverse=True
        )
        
        # Select top clients up to budget
        selected = sorted_clients[:context.selection_budget]
        
        return selected

class DiversityAwareLLMSelector(LLMClientSelector):
    """Selection that balances performance and diversity"""
    
    def __init__(self, diversity_weight: float = 0.3):
        self.diversity_weight = diversity_weight
    
    def select_clients(self, context: SelectionContext, reward_scores: Dict[int, float]) -> List[int]:
        """Select clients balancing performance and diversity"""
        selected = []
        remaining = context.available_clients.copy()
        
        # First, select the best performing client
        if remaining:
            best_client = max(remaining, key=lambda x: reward_scores.get(x, 0.0))
            selected.append(best_client)
            remaining.remove(best_client)
        
        # Then select diverse clients
        while len(selected) < context.selection_budget and remaining:
            best_score = -float('inf')
            best_client = None
            
            for client in remaining:
                # Performance score
                perf_score = reward_scores.get(client, 0.0)
                
                # Diversity score (based on model_diversity and selection history)
                diversity_score = 0.0
                if client in context.recent_performance:
                    diversity_score = context.recent_performance[client].model_diversity
                
                # Combined score
                combined_score = (1 - self.diversity_weight) * perf_score + \
                               self.diversity_weight * diversity_score
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_client = client
            
            if best_client is not None:
                selected.append(best_client)
                remaining.remove(best_client)
            else:
                break
        
        return selected

class OllamaLLMSelector(LLMClientSelector):
    """LLM-based selector using Ollama (Llama3) for intelligent client selection"""
    
    def __init__(self, model_name: str = "llama3:8b", base_url: str = "http://localhost:11434"):
        self.ollama_client = OllamaLLMClient(model_name, base_url)
        self.fallback_selector = AdaptiveLLMSelector()  # Fallback if Ollama is not available
        self.logger = logging.getLogger(__name__)
        
        # Check if Ollama is available
        if not self.ollama_client.is_available():
            self.logger.warning("Ollama service not available, will use fallback selector")
            self.use_ollama = False
        else:
            self.logger.info(f"Ollama service available with model: {model_name}")
            self.use_ollama = True
    
    def select_clients(self, context: SelectionContext, reward_scores: Dict[int, float]) -> List[int]:
        """Select clients using Llama3 via Ollama"""
        
        if not self.use_ollama:
            self.logger.info("Using fallback selector (Ollama not available)")
            return self.fallback_selector.select_clients(context, reward_scores)
        
        try:
            # Use Llama3 for selection
            selected_clients = self.ollama_client.select_clients_llm(
                context, reward_scores, context.available_clients
            )
            
            self.logger.info(f"Llama3 selected clients: {selected_clients}")
            return selected_clients
            
        except Exception as e:
            self.logger.error(f"Error in Ollama selection: {e}")
            self.logger.info("Falling back to adaptive selector")
            return self.fallback_selector.select_clients(context, reward_scores)

class AdaptiveLLMSelector(LLMClientSelector):
    """Adaptive selection that learns from past performance"""
    
    def __init__(self, exploration_rate: float = 0.1, learning_rate: float = 0.01):
        self.exploration_rate = exploration_rate
        self.learning_rate = learning_rate
        self.selection_history = []
        self.performance_history = []
    
    def select_clients(self, context: SelectionContext, reward_scores: Dict[int, float]) -> List[int]:
        """Adaptive selection with exploration and exploitation"""
        # Calculate selection probabilities
        probabilities = self._calculate_selection_probabilities(context, reward_scores)
        
        # Sample clients based on probabilities
        selected = []
        remaining = context.available_clients.copy()
        
        for _ in range(min(context.selection_budget, len(remaining))):
            if not remaining:
                break
                
            # Sample client based on probabilities
            client_probs = [probabilities.get(client, 0.0) for client in remaining]
            if sum(client_probs) == 0:
                # Fallback to uniform selection
                client_probs = [1.0 / len(remaining)] * len(remaining)
            
            # Normalize probabilities
            total_prob = sum(client_probs)
            client_probs = [p / total_prob for p in client_probs]
            
            # Sample client
            selected_client = np.random.choice(remaining, p=client_probs)
            selected.append(selected_client)
            remaining.remove(selected_client)
        
        return selected
    
    def _calculate_selection_probabilities(self, context: SelectionContext, 
                                         reward_scores: Dict[int, float]) -> Dict[int, float]:
        """Calculate selection probabilities for each client"""
        probabilities = {}
        
        for client in context.available_clients:
            # Base probability from reward score
            base_prob = reward_scores.get(client, 0.0)
            
            # Exploration bonus
            exploration_bonus = self.exploration_rate * (1.0 / len(context.available_clients))
            
            # Historical performance adjustment
            if client in context.recent_performance:
                metrics = context.recent_performance[client]
                # Reduce probability for frequently selected clients
                frequency_penalty = 0.1 * (metrics.selection_count / (context.current_round + 1))
                # Increase probability for clients not selected recently
                recency_bonus = 0.1 * min((context.current_round - metrics.last_selected) / 10.0, 1.0)
                
                base_prob = base_prob - frequency_penalty + recency_bonus
            
            probabilities[client] = max(0.0, base_prob + exploration_bonus)
        
        return probabilities

class LLMClientSelectionSystem:
    """
    Main system for LLM-based client selection in federated learning
    """
    
    def __init__(self, 
                 n_clients: int,
                 selection_strategy: str = "adaptive",
                 reward_model_path: Optional[str] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the LLM client selection system
        
        Args:
            n_clients: Total number of clients
            selection_strategy: Strategy to use ("greedy", "diversity", "adaptive")
            reward_model_path: Path to pre-trained reward model
            logger: Logger instance
        """
        self.n_clients = n_clients
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize reward model
        self.reward_model = RewardModel()
        if reward_model_path and os.path.exists(reward_model_path):
            self.reward_model.load_state_dict(torch.load(reward_model_path))
            self.logger.info(f"Loaded reward model from {reward_model_path}")
        
        # Initialize selection strategy
        self.selection_strategy = self._create_selector(selection_strategy)
        
        # Performance tracking
        self.client_metrics_history = []
        self.selection_history = []
        self.global_performance_history = []
        
        # Training parameters for reward model
        self.reward_optimizer = torch.optim.Adam(self.reward_model.parameters(), lr=0.001)
        self.reward_criterion = nn.MSELoss()
    
    def _create_selector(self, strategy: str) -> LLMClientSelector:
        """Create selector based on strategy name"""
        if strategy == "greedy":
            return GreedyLLMSelector()
        elif strategy == "diversity":
            return DiversityAwareLLMSelector()
        elif strategy == "adaptive":
            return AdaptiveLLMSelector()
        elif strategy == "ollama":
            return OllamaLLMSelector()
        elif strategy == "llama3":
            return OllamaLLMSelector(model_name="llama3:8b")
        else:
            raise ValueError(f"Unknown selection strategy: {strategy}")
    
    def update_client_metrics(self, 
                            client_id: int,
                            loss: float,
                            accuracy: float,
                            data_quality: float = 1.0,
                            performance_trend: float = 0.0,
                            resource_availability: float = 1.0,
                            model_diversity: float = 0.5):
        """Update metrics for a specific client"""
        current_round = len(self.client_metrics_history)
        
        # Get previous metrics if available
        prev_metrics = None
        if self.client_metrics_history and client_id in self.client_metrics_history[-1]:
            prev_metrics = self.client_metrics_history[-1][client_id]
        
        # Create new metrics
        metrics = ClientMetrics(
            client_id=client_id,
            loss=loss,
            accuracy=accuracy,
            data_quality=data_quality,
            communication_round=current_round,
            last_selected=prev_metrics.last_selected if prev_metrics else current_round,
            selection_count=prev_metrics.selection_count if prev_metrics else 0,
            performance_trend=performance_trend,
            resource_availability=resource_availability,
            model_diversity=model_diversity
        )
        
        # Update metrics history
        if not self.client_metrics_history or len(self.client_metrics_history[-1]) == 0:
            self.client_metrics_history.append({})
        
        self.client_metrics_history[-1][client_id] = metrics
    
    def calculate_reward_scores(self, current_round: int) -> Dict[int, float]:
        """Calculate reward scores for all clients"""
        reward_scores = {}
        
        if not self.client_metrics_history:
            # First round: uniform scores
            return {i: 1.0 / self.n_clients for i in range(self.n_clients)}
        
        # Get latest metrics
        latest_metrics = self.client_metrics_history[-1]
        
        for client_id in range(self.n_clients):
            if client_id in latest_metrics:
                metrics = latest_metrics[client_id]
                
                # Extract features
                features = self.reward_model.extract_features(metrics)
                
                # Calculate reward score
                with torch.no_grad():
                    reward_score = self.reward_model(features.unsqueeze(0)).item()
                
                reward_scores[client_id] = reward_score
            else:
                # Default score for clients without metrics
                reward_scores[client_id] = 0.1
        
        return reward_scores
    
    def select_clients(self, 
                      current_round: int,
                      selection_budget: int,
                      available_clients: Optional[List[int]] = None,
                      global_performance: float = 0.0) -> List[int]:
        """
        Select clients for the current round
        
        Args:
            current_round: Current communication round
            selection_budget: Number of clients to select
            available_clients: List of available clients (if None, use all)
            global_performance: Current global model performance
            
        Returns:
            selected_clients: List of selected client IDs
        """
        if available_clients is None:
            available_clients = list(range(self.n_clients))
        
        # Calculate reward scores
        reward_scores = self.calculate_reward_scores(current_round)
        
        # Create selection context
        context = SelectionContext(
            current_round=current_round,
            total_rounds=100,  # This could be passed as parameter
            available_clients=available_clients,
            recent_performance=self.client_metrics_history[-1] if self.client_metrics_history else {},
            global_model_performance=global_performance,
            selection_budget=selection_budget,
            diversity_requirement=0.3,
            performance_threshold=0.1
        )
        
        # Select clients using the strategy
        selected_clients = self.selection_strategy.select_clients(context, reward_scores)
        
        # Update selection history
        self.selection_history.append({
            'round': current_round,
            'selected_clients': selected_clients,
            'reward_scores': reward_scores,
            'timestamp': datetime.now().isoformat()
        })
        
        # Update last_selected for selected clients
        if self.client_metrics_history:
            for client_id in selected_clients:
                if client_id in self.client_metrics_history[-1]:
                    self.client_metrics_history[-1][client_id].last_selected = current_round
                    self.client_metrics_history[-1][client_id].selection_count += 1
        
        self.logger.info(f"Selected clients for round {current_round}: {selected_clients}")
        self.logger.info(f"Reward scores: {reward_scores}")
        
        return selected_clients
    
    def train_reward_model(self, 
                          client_metrics: List[ClientMetrics],
                          performance_feedback: List[float]):
        """
        Train the reward model based on performance feedback
        
        Args:
            client_metrics: List of client metrics
            performance_feedback: Performance feedback scores
        """
        if len(client_metrics) != len(performance_feedback):
            raise ValueError("client_metrics and performance_feedback must have same length")
        
        # Prepare training data
        features = []
        targets = []
        
        for metrics, feedback in zip(client_metrics, performance_feedback):
            feature = self.reward_model.extract_features(metrics)
            features.append(feature)
            targets.append(feedback)
        
        features = torch.stack(features)
        targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
        
        # Training step
        self.reward_optimizer.zero_grad()
        predictions = self.reward_model(features)
        loss = self.reward_criterion(predictions, targets)
        loss.backward()
        self.reward_optimizer.step()
        
        self.logger.info(f"Reward model training loss: {loss.item():.4f}")
    
    def save_reward_model(self, path: str):
        """Save the reward model"""
        torch.save(self.reward_model.state_dict(), path)
        self.logger.info(f"Reward model saved to {path}")
    
    def load_reward_model(self, path: str):
        """Load the reward model"""
        self.reward_model.load_state_dict(torch.load(path))
        self.logger.info(f"Reward model loaded from {path}")
    
    def get_selection_statistics(self) -> Dict[str, Any]:
        """Get statistics about client selection"""
        if not self.selection_history:
            return {}
        
        # Calculate selection frequency
        selection_counts = {}
        for record in self.selection_history:
            for client_id in record['selected_clients']:
                selection_counts[client_id] = selection_counts.get(client_id, 0) + 1
        
        # Calculate average reward scores
        avg_reward_scores = {}
        for record in self.selection_history:
            for client_id, score in record['reward_scores'].items():
                if client_id not in avg_reward_scores:
                    avg_reward_scores[client_id] = []
                avg_reward_scores[client_id].append(score)
        
        for client_id in avg_reward_scores:
            avg_reward_scores[client_id] = np.mean(avg_reward_scores[client_id])
        
        return {
            'selection_counts': selection_counts,
            'avg_reward_scores': avg_reward_scores,
            'total_rounds': len(self.selection_history),
            'selection_diversity': len(set().union(*[r['selected_clients'] for r in self.selection_history]))
        }

def create_llm_client_selector(n_clients: int, 
                              strategy: str = "adaptive",
                              logger: Optional[logging.Logger] = None) -> LLMClientSelectionSystem:
    """
    Factory function to create LLM client selector
    
    Args:
        n_clients: Number of clients
        strategy: Selection strategy ("greedy", "diversity", "adaptive", "ollama", "llama3")
        logger: Logger instance
        
    Returns:
        LLMClientSelectionSystem instance
    """
    return LLMClientSelectionSystem(
        n_clients=n_clients,
        selection_strategy=strategy,
        logger=logger
    )

# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create client selector
    n_clients = 8
    selector = create_llm_client_selector(n_clients, strategy="adaptive", logger=logger)
    
    # Simulate some rounds
    for round_num in range(5):
        # Update client metrics (simulated)
        for client_id in range(n_clients):
            loss = random.uniform(0.1, 2.0)
            accuracy = random.uniform(0.5, 0.95)
            data_quality = random.uniform(0.7, 1.0)
            performance_trend = random.uniform(-0.1, 0.1)
            resource_availability = random.uniform(0.8, 1.0)
            model_diversity = random.uniform(0.3, 0.8)
            
            selector.update_client_metrics(
                client_id=client_id,
                loss=loss,
                accuracy=accuracy,
                data_quality=data_quality,
                performance_trend=performance_trend,
                resource_availability=resource_availability,
                model_diversity=model_diversity
            )
        
        # Select clients
        selected = selector.select_clients(
            current_round=round_num,
            selection_budget=4,
            global_performance=random.uniform(0.7, 0.9)
        )
        
        print(f"Round {round_num}: Selected clients {selected}")
    
    # Print statistics
    stats = selector.get_selection_statistics()
    print(f"Selection statistics: {stats}")

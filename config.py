from dataclasses import dataclass

@dataclass
class GPTConfig:
    # Reduced model size since we have a smaller dataset
    block_size: int = 512  # Reduced from 1024 since TV scripts rarely need long context
    vocab_size: int = 50304  # Keep standard GPT-2 vocab
    n_layer: int = 8  # Reduced from 12
    n_head: int = 8  # Reduced from 12
    n_embd: int = 512  # Reduced from 768
    
    # Training specific parameters
    dropout: float = 0.2  # Add dropout for regularization
    learning_rate: float = 3e-4  # Slightly lower learning rate
    warmup_steps: int = 100
    max_steps: int = 2000  # Increased training steps
    grad_clip: float = 1.0
    weight_decay: float = 0.1
    batch_size: int = 16
    
    # Generation parameters
    temperature: float = 0.8  # Controls randomness in generation
    top_k: int = 40  # For top-k sampling
    top_p: float = 0.9  # For nucleus sampling

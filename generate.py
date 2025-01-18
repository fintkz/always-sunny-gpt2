import tiktoken
import torch
import re

def generate_sunny_script(
    model,
    prompt,
    max_length=200,
    num_return_sequences=3,
    device='cuda',
    config=None
):
    """Generate Always Sunny style content."""
    if config is None:
        config = model.config
    
    tokenizer = tiktoken.get_encoding('gpt2')
    
    # Add scene heading if not present
    if not any(marker in prompt.upper() for marker in ['INT.', 'EXT.']):
        prompt = "INT. PADDY'S PUB - DAY\n\n" + prompt
    
    # Tokenize and prepare input
    tokens = torch.tensor(tokenizer.encode(prompt), dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    tokens = tokens.to(device)
    
    # Track generated content for filtering duplicates
    generated_texts = []
    
    # Generation loop
    with torch.no_grad():
        for _ in range(max_length):
            logits, _ = model(tokens)
            logits = logits[:, -1, :] / config.temperature
            
            # Filter using both top-k and top-p (nucleus) sampling
            top_k_scores, top_k_indices = torch.topk(logits, config.top_k, dim=-1)
            probs = torch.nn.functional.softmax(top_k_scores, dim=-1)
            
            # Nucleus sampling
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_keep = cumsum_probs <= config.top_p
            
            # Sample from the filtered distribution
            filtered_probs = torch.where(sorted_indices_to_keep, sorted_probs, torch.zeros_like(sorted_probs))
            filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
            
            sampled_sorted_idx = torch.multinomial(filtered_probs, num_samples=1)
            sampled_top_k_idx = torch.gather(top_k_indices, -1, 
                torch.gather(sorted_indices, -1, sampled_sorted_idx))
            
            tokens = torch.cat((tokens, sampled_top_k_idx), dim=1)
    
    # Decode and clean up the generated texts
    for i in range(num_return_sequences):
        text = tokenizer.decode(tokens[i].tolist())
        
        # Clean up generation
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1\n\n\2', text)  # Add proper line breaks
        text = re.sub(r'([A-Z]{2,}:)', r'\n\1', text)  # Format character names
        
        generated_texts.append(text)
    
    return generated_texts

def generate_examples(model, device='cuda'):
    """Generate example scripts with different prompts."""
    prompts = [
        "INT. PADDY'S PUB - DAY\n\nCharlie bursts into the bar, holding a rat trap.",
        "DENNIS: Guys, I've got a foolproof plan.",
        "MAC: I've been doing some ocular pat-downs and",
        "DEE: Oh my god, you guys are not gonna believe this.",
        "FRANK: So anyway, I started blasting"
    ]
    
    print("Generating Always Sunny style content...")
    print("=" * 60)
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}\n")
        generations = generate_sunny_script(
            model,
            prompt,
            max_length=150,
            num_return_sequences=1,
            device=device
        )
        print(generations[0])
        print("\n" + "=" * 60)

if __name__ == '__main__':
    # Test generation
    from model import GPT
    from config import GPTConfig
    
    model = GPT(GPTConfig())
    model.load_state_dict(torch.load('model.pth'))
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    generate_examples(model)

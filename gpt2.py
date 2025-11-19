# We are going to use the GPT-2 model to generate a proof that there are infinitely many primes that are congruent to 1 mod 4.
import threading

from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch

def main():
    # 1. Initialize the base model architecture
    model_name = "gpt2-medium"
    print(f"Initializing base model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # 2. Load your trained weights
    checkpoint_path = "checkpoints/step_600.pt"
    print(f"Overwriting weights with checkpoint: {checkpoint_path}...")
    
    # Load the dictionary to CPU first to avoid memory issues
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    
    # Apply the weights. strict=False ignores the extra 'v_head' (value head) 
    # that PPO adds but isn't part of the standard GPT-2 inference model.
    model.load_state_dict(state_dict, strict=False)
    print("✓ Checkpoint loaded successfully.")

    # 3. Move to GPU and run inference
    model.to("cuda")
    model.eval()

    input_text = "Despite much style flash and glitter, this french"
    
    # 1. Use tokenizer(...) instead of tokenizer.encode(...)
    #    This returns a dictionary containing 'input_ids' AND 'attention_mask'
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    
    print(f"Generating text for prompt: '{input_text}'")
    
    # 2. Pass both to generate
    output = model.generate(
        **inputs,  # Unpacks input_ids and attention_mask
        max_length=64, 
        num_return_sequences=1, 
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id # Explicitly set pad_token_id to silence that specific warning too
    )
    print("-" * 40)
    print(tokenizer.decode(output[0], skip_special_tokens=True))
    print("-" * 40)

if __name__ == "__main__":
    main()

from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead
import torch

def main():
    device = torch.device("cuda")
    print(f"Using device {device}")

    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    policy = AutoModelForCausalLMWithValueHead.from_pretrained(model_name,torch_dtype=torch.bfloat16,device_map={"": 0})
    policy.gradient_checkpointing_enable()
    total_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f'total_params = {total_params}')

    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name,torch_dtype=torch.bfloat16,device_map={"": 0})
    ref_model.requires_grad_(False)
    total_params = sum(p.numel() for p in ref_model.parameters() if p.requires_grad)
    print(f'total_params = {total_params}')


    prompt = """Find the global maximum value of f and global minimum value of f for f (x) = x^3 - 3x^2 + 20
for x ∈ [−1, 3], if they exist"""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer([text], return_tensors="pt").to(device)

    outputs = policy.generate(**inputs, max_new_tokens=2000)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(decoded.split(prompt)[-1].strip())

if __name__ == "__main__":
    main()

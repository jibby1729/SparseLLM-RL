# We are going to use the GPT-2 model to generate a proof that there are infinitely many primes that are congruent to 1 mod 4.
import threading

from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch

def main():
    device = torch.device("cuda")
    print(f"Using device {device}")
    model_name = "gpt2-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16) 
    print(f"Loaded model {model_name} with {model.num_parameters()} parameters")

    prompt = (
        "There are infinitely many primes, and the proof begins with the following step: "
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_thread = threading.Thread(
        target=model.generate,
        kwargs=dict(
            **inputs,
            max_length= 512,
            streamer=streamer,
        ),
    )
    generation_thread.start()

    generated_tokens = []
    for token in streamer:
        print(token, end="", flush=True)
        generated_tokens.append(token)

    generation_thread.join()
    print()

if __name__ == "__main__":
    main()

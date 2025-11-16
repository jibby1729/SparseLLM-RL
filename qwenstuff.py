import threading

from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

model_name = "Qwen/Qwen2.5-Math-1.5B"
device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def main():

    prompt = (
        "Please reason step by step and put your final answer within \\boxed{}.\n"
        "Write a short proof that there are infinitely many primes that are congruent to 1 mod 4, "
        "Assume there are finitely many such primes and let p_1, p_2, \cdots, p_n be the list of all such primes."
        "Consider the number 4*p_1^2 p_2^2 \cdots p_n^2 + 1. This number is congruent to "
    )

    model_inputs = tokenizer(prompt, return_tensors="pt").to(device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_thread = threading.Thread(
        target=model.generate,
        kwargs=dict(
            **model_inputs,
            max_new_tokens=1024,
            streamer=streamer
        ),
    )
    generation_thread.start()

    response_tokens = []
    for token in streamer:
        print(token, end="", flush=True)
        response_tokens.append(token)

    generation_thread.join()
    print()  # final newline after streaming output
    return "".join(response_tokens)

if __name__ == "__main__":
    response = main()
    print(response)
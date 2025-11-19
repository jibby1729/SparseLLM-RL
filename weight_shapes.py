from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("gpt2-medium")
w1 = model.transformer.h[1].attn.c_attn.weight   # torch.Tensor
print('Attention Input weight shape:', w1.shape)  

w2 = model.transformer.h[2].attn.c_proj.weight
print('Attention Projection weight shape:', w2.shape)  

w3 = model.transformer.h[0].mlp.c_fc.weight
print('MLP FC weight shape:', w3.shape)  
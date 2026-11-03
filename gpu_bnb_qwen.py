import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 1. Model Name
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# 2. BitsAndBites 4-bit quantize configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",                # NormalFloat 4 (best quality)
    bnb_4bit_compute_dtype=torch.float16,     # GPU computation speed-kku FP16
    bnb_4bit_use_double_quant=True            # Extra VRAM savings
)

# 3. Load Model and Tokenizer straight into GPU
print("----[ Loading model in 4-bit on GPU ]----")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"                         # Direct-ah GPU (cuda:0)-kku map pannum
)

print("\n" + "="*50)
print("🤖 Qwen 4-bit GPU Chatbot Ready! (Type '/bye' to quit)")
print("="*50,"\n")

messages = [{"role": "system", "content": "you are a friendly ai bot called 'MINI'."}]

while True:
    question = input("YOU: ")

    if question.lower() in ["/bye"]:
        break

    if not question.strip():
        continue

    if question.lower() == "/clear":
        messages = [{"role": "system", "content": "you are a friendly ai bot called 'MINI'."}]
        print(messages)
        continue
    
    messages.append({"role": "user", "content": question})

    sttime = time.time()
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 5. Move inputs to GPU (cuda)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # 6. Generate Response
    print("AI: ", end="", flush=True)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=256)
    sptime = time.time()
    # 7. Decode Output
    answer = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    print(answer)
    print("VRAM USED:", f"{torch.cuda.memory_allocated() / (1024**2):.2f} MB")
    print(f"Time Taken: {(sptime-sttime): 3.2f}")
    print("="*60)
    messages.append({"role": "assistant", "content": answer})
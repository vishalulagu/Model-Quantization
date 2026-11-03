
import time
from transformers import AutoTokenizer
from optimum.quanto import QuantizedModelForCausalLM

# The folder where quantize_qwen.py saved the small model.
SAVED_FOLDER = "./qwen-int8"

# --- Step 1: load the small model and tokenizer from the local folder ---
print("Loading the small model...")
model = QuantizedModelForCausalLM.from_pretrained(SAVED_FOLDER)
tokenizer = AutoTokenizer.from_pretrained(SAVED_FOLDER)
print("\n" + "="*50)
print("🤖 Qwen 8-bit CPU Chatbot Ready! (Type '/bye' to quit)")
print("="*50,"\n")

messages = [{"role": "system", "content": "You are a helpful ai bot called MINI."}]

while True:
    question = input("YOU: ")

    if question.lower() in ["/bye", "/exit"]:
        break

    if not question.strip():
        continue
    sttime = time.time()
    messages.append({"role": "user", "content": question})
    # Qwen is a chat model, so we put our question in chat format.
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")

    # --- Step 3: let the model generate an answer (the OUTPUT) ---
    print("Thinking...")
    output = model.generate(**inputs, max_new_tokens=200)

    # Keep only the new words the model added (remove our question).
    answer = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    sptime = time.time()
    # --- Step 4: show the output ---
    print(f"OUTPUT: {answer}")
    print(f"TimeTaken: {(sptime-sttime):3.2f}\n")
    messages.append({"role": "assistant", "content": answer})
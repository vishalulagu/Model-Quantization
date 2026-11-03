
from transformers import AutoModelForCausalLM, AutoTokenizer 
from optimum.quanto import QuantizedModelForCausalLM, qint8 

# The model we want and the folder where we will save the small version.
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
SAVE_FOLDER = "./qwen-int8"

# --- Step 1: download the original (big) model + its tokenizer ---
print("Downloading the model... (this can take a few minutes the first time)")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# --- Step 2: quantise the model to int8 (8-bit numbers instead of 32-bit) ---
print("Quantizing the model to int8...")
small_model = QuantizedModelForCausalLM.quantize(model, weights=qint8)

# --- Step 3: save the small model and tokenizer into a local folder ---
print("Saving the small model to:", SAVE_FOLDER)
small_model.save_pretrained(SAVE_FOLDER)
tokenizer.save_pretrained(SAVE_FOLDER)

print("Model Quantized Successfull!...👍")
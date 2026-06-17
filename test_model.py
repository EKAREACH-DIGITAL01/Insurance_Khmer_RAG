from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "SeaLLMs/SeaLLMs-v3-7B-Chat"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
)

print("SUCCESS")

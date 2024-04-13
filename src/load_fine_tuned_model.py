import os 
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from constants.paths import model_weights_directory
base_model_id = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(
          base_model_id, trust_remote_code=True, quantization_config=bnb_config, device_map={"": 0})
# adapter = "/content/phi2-results2/checkpoint-100"
adapter = os.path.join(model_weights_directory, "checkpoint-100")
model = PeftModel.from_pretrained(model, adapter)

def query_solver(query, tokenizer=  tokenizer , model = model):
  model_inputs = tokenizer(query, return_tensors="pt").to("cuda:0")
  output = model.generate(**model_inputs, max_length=500, no_repeat_ngram_size=10, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)[0]
  return tokenizer.decode(output, skip_special_tokens=False)

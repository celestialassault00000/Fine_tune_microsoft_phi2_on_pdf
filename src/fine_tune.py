from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, prepare_model_for_kbit_training
import torch

if __name__ == "__main__":
    base_model_id = "microsoft/phi-2"
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_eos_token=True, use_fast=True, max_length=250)
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token
    compute_dtype = getattr(torch, "float16")
    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,)
    model = AutoModelForCausalLM.from_pretrained(
              base_model_id, trust_remote_code=True, quantization_config=bnb_config, revision="refs/pr/23", device_map={"": 0}, torch_dtype="auto", flash_attn=True, flash_rotary=True, fused_dense=True)
    
    model = prepare_model_for_kbit_training(model)
    
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.5,
        r=32,
        target_modules=['k_proj', 'q_proj', 'v_proj', 'fc1', 'fc2'],
        bias="none",
        task_type="CAUSAL_LM")
    
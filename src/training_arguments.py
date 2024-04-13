from transformers import TrainingArguments
from constants.paths import model_weights_directory
training_arguments = TrainingArguments(
        output_dir=model_weights_directory,
        save_strategy="epoch",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=12,
        log_level="debug",
        save_steps=100,
        logging_steps=25,
        learning_rate=1e-4,
        eval_steps=50,
        optim='paged_adamw_8bit',
        fp16=True, #change to bf16 if are using an Ampere GPU
        num_train_epochs=1,
        max_steps=100,
        warmup_steps=100,
        lr_scheduler_type="linear",
        seed=42)
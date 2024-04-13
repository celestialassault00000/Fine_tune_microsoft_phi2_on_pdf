from trl import SFTTrainer
from src.create_and_load_dataset import load_dataset
from src.load_fine_tuned_model import model, peft_config , tokenizer
from src.training_arguments import training_arguments
dataset = load_dataset
def training_function(dataset = dataset, model = model, peft_config = peft_config, tokenizer = tokenizer, args = training_arguments):
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=200,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False)
    trainer.train()
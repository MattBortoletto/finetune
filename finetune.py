"""
CUDA_VISIBLE_DEVICES=0 python finetune.py --num_train_epochs 1 
"""
import os
import random 
import argparse
from datetime import datetime
from termcolor import cprint 
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
    logging as hf_logging,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

from utils import set_random_seed, load_tomssi, format_prompt_tomssi, format_prompt_tomssi_reasoning

hf_logging.set_verbosity_error()

hf_token = os.getenv("HF_TOKEN")
if hf_token is None:
    raise ValueError("Hugging Face token not found. Ensure HF_TOKEN is set.")


class DebugCallback(TrainerCallback):
    def __init__(self, tokenizer, eval_dataset):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset

    def on_evaluate(self, args, state, control, **kwargs):
        print("\n--- Debug Mode: Model Prediction Sample ---")
        
        # Access the evaluation dataset from the trainer's kwargs
        eval_dataset = self.eval_dataset
        
        # Select a random example from the evaluation dataset
        sample_idx = random.randint(0, len(eval_dataset) - 1)
        sample = eval_dataset[sample_idx]
        
        prompt = sample["prompt"]
        true_completion = sample["completion"][0]["content"].strip()
        
        generator = pipeline(
            "text-generation",
            model=kwargs['model'],
            tokenizer=self.tokenizer,
            max_new_tokens=50,
            do_sample=False, # greedy search, select the most likely token 
        )
        
        generated_text = generator(prompt)[0]["generated_text"]
        answer = generated_text[1]["content"].strip()

        #print(f"**Prompt:**\n{prompt_content.strip()}")
        #print("-" * 20)
        print(f"**True Completion:**\n{true_completion.strip()}")
        print("-" * 20)
        print(f"**Model Prediction:**\n{answer}")
        print("-" * 20)
        print("--- End of Debug Print ---")
        breakpoint()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Model to load from Huggingface")
    parser.add_argument("--save_log", action="store_true", help="Save logs")
    parser.add_argument("--verbose", action="store_true", help="Prints")
    parser.add_argument("--dataset", type=str, required=True, default="tomssi-reasoning", choices=["tomssi", "tomssi-reasoning"], help="Prints")
    
    # Fine-Tuning Arguments
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "int4", "int8"], help="Training precision")
    parser.add_argument("--peft", action="store_true", help="Use LoRA/PEFT for training")

    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha scaling factor")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout probability")
    parser.add_argument("--max_steps", type=int, default=-1, help="Number of training epochs")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--optim", type=str, default="paged_adamw_32bit", help="Optimizer")

    # Debug
    parser.add_argument("--debug", action="store_true", help="Enables debug printing of prompts and predictions.")

    args = parser.parse_args()

    set_random_seed(args.seed)
    
    current_time = datetime.now()
    formatted_time = current_time.strftime("%b-%d-%Y-%H-%M-%S")
    save_folder = f"./ft_{args.model_name.split("/")[-1]}_{formatted_time}"

    train_data, eval_data, _ = load_tomssi(data_dir=f"data/{args.dataset}", split_data=True, print_ids=True)
    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)
    if args.dataset == "tomssi":
        train_dataset = train_dataset.map(format_prompt_tomssi, remove_columns=['full_prompt', 'correct_answer', 'idx', 'system_prompt', 'state_prompt', 'grid_prompt', 'info_event_attitude_prompt', 'question_prompt', 'question', 'choices'])
        eval_dataset = eval_dataset.map(format_prompt_tomssi,   remove_columns=['full_prompt', 'correct_answer', 'idx', 'system_prompt', 'state_prompt', 'grid_prompt', 'info_event_attitude_prompt', 'question_prompt', 'question', 'choices'])
    elif args.dataset == "tomssi-reasoning":
        train_dataset = train_dataset.map(format_prompt_tomssi_reasoning, remove_columns=['reasoning', 'full_prompt', 'correct_answer', 'idx', 'system_prompt', 'state_prompt', 'grid_prompt', 'info_event_attitude_prompt', 'question_prompt', 'question', 'choices'])
        eval_dataset = eval_dataset.map(format_prompt_tomssi_reasoning,   remove_columns=['reasoning', 'full_prompt', 'correct_answer', 'idx', 'system_prompt', 'state_prompt', 'grid_prompt', 'info_event_attitude_prompt', 'question_prompt', 'question', 'choices'])
    else:
        raise ValueError

    print(f"\n###############\nExample of completion:\n")
    cprint(train_dataset[1]['completion'][0]['content'], "yellow")
    print("\n###############\n")

    # Determine training mode based on arguments
    lora_config = None
    bnb_config = None
    use_peft = args.peft
    use_quantization = args.precision in ["int4", "int8"]

    if use_quantization:
        print(f"Using {args.precision} quantization.")
        # Configure QLoRA and quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, # Load the model in 4-bits
            bnb_4bit_quant_type="nf4", # Use a special 4-bit data type for weights initialised from N(0,1)
            bnb_4bit_compute_dtype=torch.bfloat16, # For faster computations
            bnb_4bit_use_double_quant=True, # Use a nested quantization scheme to quantize the already quantized weights
        )

    if use_peft:
        print(f"Using LoRA.")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

    # Load model and tokenizer
    print(f"Loading model: {args.model_name}.")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Llama models often prefer 'right' padding
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token,
        torch_dtype=torch.bfloat16 if not use_quantization else None
    )
    if use_peft:
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        print(f"PEFT trainable parameters: {model.print_trainable_parameters()}")

    # Define Training Arguments
    training_args = SFTConfig(
        output_dir="./fine_tuned_model",
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs, # This argument will be ignored if max_steps is set
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        optim=args.optim,
        save_strategy="epoch",
        logging_steps=10,
        bf16=True if args.precision == "bf16" else False,
        fp16=True if args.precision == "fp16" else False,
        report_to="tensorboard" if args.save_log else None,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, # safer way of doing gradient checkpointing 
        max_length=1024,
        packing=True, # Packing reduces padding by merging several sequences in one row when possible
        eval_strategy="steps", # Evaluate model at regular intervals
        eval_steps=50, # Run model evaluation every 50 steps
    )

    # 4. Initialize and Run SFTTrainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        args=training_args,
    )

    # Add the custom callback if debug mode is enabled
    if args.debug:
        print("Debug mode enabled. Adding custom callback.")
        debug_eval_dataset = Dataset.from_list(eval_data)
        debug_eval_dataset = debug_eval_dataset.map(format_prompt_tomssi, remove_columns=['full_prompt', 'correct_answer', 'idx', 'system_prompt', 'state_prompt', 'grid_prompt', 'info_event_attitude_prompt', 'question_prompt', 'question', 'choices'])
        trainer.add_callback(DebugCallback(tokenizer, debug_eval_dataset))

    # Train the model
    print("Starting training...")
    trainer.train()
    print("Training complete!")

    # Save the final model and tokenizer
    trainer.save_model(save_folder)

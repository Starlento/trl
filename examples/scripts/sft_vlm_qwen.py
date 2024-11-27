"""# Tested on single GPU
accelerate launch `
    --config_file "examples/accelerate_configs/single_gpu.yaml" `
    examples/scripts/sft_vlm_qwen.py `
    --dataset_name "./downloads/llava-instruct-mix-vsft" `
    --model_name_or_path "./downloads/Qwen2-VL-2B-Instruct" `
    --per_device_train_batch_size 1 `
    --gradient_accumulation_steps 1 `
    --output_dir "./downloads/sft-Qwen2-VL-2B-Instruct" `
    --bf16 `
    --torch_dtype bfloat16 `
    --gradient_checkpointing
"""

import torch
from datasets import load_dataset
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, LlavaForConditionalGeneration

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()
    
    # print out the args
    # print(f"script_args: {script_args}")
    # print(f"training_args: {training_args}")
    # print(f"model_config: {model_config}")
    
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    ################
    # Model, Tokenizer & Processor
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    processor = AutoProcessor.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code
    )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, **model_kwargs
    )
    
    # freeze model.visual
    for param in model.visual.parameters():
        param.requires_grad = False
    
    # freeze model.lm_head
    for param in model.lm_head.parameters():
        param.requires_grad = False
        
    # # freeze model.model
    # for param in model.model.parameters():
    #     param.requires_grad = False
        
    # print number of trainable parameters in Million
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {trainable_params/1e6:.2f}M")

    ################
    # Create a data collator to encode text and image pairs
    ################
    def collate_fn(examples):
        # Get the texts and images, and apply the chat template
        texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
        images = [example["images"] for example in examples]
        if isinstance(model, LlavaForConditionalGeneration):
            # LLava1.5 does not support multiple images
            images = [image[0] for image in images]

        # Tokenize the texts and process the images
        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100  #
        # Ignore the image token index in the loss computation (model specific)
        image_token_id = 151655
        labels[labels == image_token_id] = -100
        batch["labels"] = labels

        return batch

    ################
    # Dataset
    ################
    dataset = load_dataset(script_args.dataset_name)

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=processor.tokenizer,
        peft_config=get_peft_config(model_config),
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    # if training_args.push_to_hub:
    #     trainer.push_to_hub(dataset_name=script_args.dataset_name)
    #     if trainer.accelerator.is_main_process:
    #         processor.push_to_hub(training_args.hub_model_id)

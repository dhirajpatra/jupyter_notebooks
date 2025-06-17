
from fastapi import FastAPI, Request
from pydantic import BaseModel
import torch
import os

app = FastAPI()

@app.get("/")
def root():
    return {"message": "FastAPI server for fine-tuning CodeLLaMA"}

@app.post("/check-gpu")
def check_gpu():
    if torch.cuda.is_available():
        return {
            "cuda_available": True,
            "device_name": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda
        }
    return {"cuda_available": False}


# Original notebook logic

# Check CUDA version first
!nvcc --version

# For CUDA 12.1+ (more likely on your server)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Core libraries
pip install transformers
pip install datasets
pip install peft
pip install trl
pip install bitsandbytes
pip install accelerate

# Flash attention (improved installation)
pip install packaging wheel
pip install flash-attn --no-build-isolation

# Tokenization and data handling
pip install sentencepiece
pip install protobuf

# Evaluation metrics
pip install evaluate
pip install rouge-score
pip install sacrebleu

# Monitoring options
pip install wandb
pip install tensorboard
pip install pynvml
pip install psutil

# TensorRT optimization
pip install tensorrt

import torch
import os

# Ensure GPU is available
assert torch.cuda.is_available(), "CUDA GPU not available."
device = torch.device("cuda")

# Print GPU info for verification
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# GPU optimization settings for RTX 4090
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,roundup_power2_divisions:16"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Optimize for RTX 4090
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for faster training
torch.backends.cudnn.allow_tf32 = True

# Clear cache at start
torch.cuda.empty_cache()

# Set memory fraction to avoid OOM (optional - adjust based on your needs)
# torch.cuda.set_per_process_memory_fraction(0.95)

#!/usr/bin/env python3
"""
Optimized Fine-tuning Pipeline for CodeLlama COBOL to Python Conversion
Using NVIDIA AI Software Stack (CUDA, cuDNN, TensorRT, PyTorch)
"""
import os
import gc
import json
import warnings
from typing import Dict, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    CodeLlamaTokenizer,  # Specific for CodeLlama
    LlamaForCausalLM     # Specific for CodeLlama
)

from datasets import load_dataset, Dataset, concatenate_datasets
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer

# Monitoring and optimization
import wandb
import tensorrt as trt
import pynvml
from evaluate import load as load_metric

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# NVIDIA optimizations for RTX 4090
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,roundup_power2_divisions:16"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Async kernel launches for speed

# RTX 4090 specific optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True  # Ada Lovelace TF32 support
torch.backends.cudnn.allow_tf32 = True

# Model configuration
model_name = "codellama/CodeLlama-7B-Instruct-hf"
output_model = "CodeLlama-7B-Instruct-COBOL-to-Python"

# Training configuration
max_seq_length = 2048  # Adjust based on your COBOL/Python code lengths
batch_size = 4         # Start with 4, can increase with 24GB VRAM
gradient_accumulation_steps = 4  # Effective batch size = 16

print(f"Using model: {model_name}")
print(f"Output will be saved as: {output_model}")
print(f"Max sequence length: {max_seq_length}")

%%capture
# Core packages for fine-tuning
%pip install -U -q transformers datasets peft accelerate huggingface_hub trl

# Quantization and optimization
%pip install -U -q bitsandbytes optimum auto-gptq

# Memory and attention optimization
%pip install -U -q xformers --no-deps

# Flash attention for better performance (if not already installed)
%pip install -U -q flash-attn --no-build-isolation

# Evaluation metrics for code conversion
%pip install -U -q evaluate rouge-score sacrebleu

class OptimizedCodeLlamaFineTuner:
    """
    A comprehensive class designed for efficient fine-tuning of CodeLlama models,
    specifically optimized for tasks like COBOL to Python code translation.

    This class leverages cutting-edge techniques such as Parameter-Efficient Fine-Tuning
    (PEFT) using LoRA (Low-Rank Adaptation) and 4-bit quantization with bitsandbytes
    to enable training large language models on consumer-grade GPUs or environments
    with limited memory.

    Key Features:
    -   **Memory-Efficient Model Loading**: Supports loading CodeLlama models in 4-bit
        quantization (`bnb.nn.Linear4bit`) with configurable double quantization
        and compute data types (e.g., bfloat16 for RTX 40 series GPUs).
    -   **LoRA Integration**: Seamlessly configures and applies LoRA adapters to
        target specific linear layers (e.g., query, key, value projections) for
        efficient fine-tuning without modifying the full model weights.
    -   **Automated Target Module Detection**: Can automatically identify all
        4-bit linear layers for LoRA application if not explicitly specified.
    -   **Dataset Handling**: Facilitates loading and preprocessing of custom datasets,
        supporting instruction-tuning formats (e.g., "### Instruction:\n...### Response:\n").
    -   **Optimized Training Arguments**: Sets up `transformers.TrainingArguments`
        with best practices for memory efficiency (gradient accumulation, gradient
        checkpointing, paged optimizers, bfloat16/fp16 precision, mixed-precision training).
    -   **Supervised Fine-Tuning (SFT)**: Utilizes `trl.SFTTrainer` for streamlined
        supervised fine-tuning, handling data collators, tokenization, and training loops.
    -   **Flexible Evaluation**: Provides methods for generating predictions from the
        fine-tuned model and calculating key evaluation metrics like Exact Match Accuracy
        and Average Similarity Score (using SequenceMatcher).
    -   **Model Management**: Supports saving LoRA adapters, merging them with the
        base model for a standalone full model, and pushing the final model/tokenizer
        to the Hugging Face Hub for easy sharing and deployment.
    -   **CUDA Memory Optimization**: Includes environment variable settings to help
        avoid CUDA memory fragmentation, improving stability during training.

    Usage Workflow:
    1.  **Initialization**: Instantiate `OptimizedCodeLlamaFineTuner` with a comprehensive
        configuration dictionary (`config_dict`) specifying model, LoRA, and training parameters.
    2.  **Load Model & Tokenizer**: Call `load_model_and_tokenizer()` to prepare the base model.
    3.  **Setup LoRA**: Invoke `setup_lora()` to apply PEFT adapters to the model.
    4.  **Load Datasets**: Use `load_your_specific_datasets()` to prepare your training and
        evaluation data.
    5.  **Create Trainer**: Call `create_and_run_trainer()` to get an initialized
        `SFTTrainer` instance.
    6.  **Train**: Execute `trainer.train()` to start the fine-tuning process.
    7.  **Evaluate**: Use `evaluate_model()` to generate predictions and
        `calculate_and_print_metrics()` to assess performance.
    8.  **Merge & Save**: Merge the LoRA adapters into the base model using `PeftModel.from_pretrained`
        and `merge_and_unload()`, then save the final model with `save_pretrained()`
        or push to the Hugging Face Hub with `push_to_hub()`.

    Parameters:
        config (dict): A dictionary containing all necessary configuration parameters
                       for model loading, LoRA setup, and training. Example parameters
                       include `model_name`, `load_in_4bit`, `lora_r`, `lora_alpha`,
                       `num_train_epochs`, `per_device_train_batch_size`, `output_dir`, etc.
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_nvidia_environment()
        self.tokenizer = None
        self.model = None
        self.datasets = []  # Support for multiple datasets
        self.combined_dataset = None
        
    def setup_nvidia_environment(self):
        """Setup NVIDIA environment and check GPU capabilities"""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available!")
            
        # Initialize NVML for GPU monitoring
        pynvml.nvmlInit()
        gpu_count = pynvml.nvmlDeviceGetCount()
        
        print(f"NVIDIA Setup:")
        print(f"   - CUDA Version: {torch.version.cuda}")
        print(f"   - cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"   - Available GPUs: {gpu_count}")
        
        for i in range(gpu_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle).decode()
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            print(f"   - GPU {i}: {name} ({memory.total // 1024**3} GB)")
            
        # Enable TensorFloat-32 for RTX 4090 (Ada Lovelace)
        if torch.cuda.get_device_capability()[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("   - TensorFloat-32 enabled for RTX 4090")
            
    def load_model_and_tokenizer(self):
        """Load CodeLlama 7B with optimized quantization"""
        
        # BitsAndBytesConfig for 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        
        print("Loading CodeLlama tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="right",
            add_eos_token=True,  # Important for code generation
            add_bos_token=True
        )
        
        # Set special tokens for code tasks
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            
        print("Loading CodeLlama model with 4-bit quantization...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2" if self.config.get("use_flash_attention", True) else "eager"
        )
        
        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        print(f"CodeLlama model loaded on: {self.model.device}")
        
    def setup_lora(self):
        """Setup LoRA configuration for CodeLlama fine-tuning"""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.config.get("lora_r", 16),
            lora_alpha=self.config.get("lora_alpha", 32),
            lora_dropout=self.config.get("lora_dropout", 0.1),
            # CodeLlama specific target modules
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        print(f"LoRA Configuration:")
        print(f"   - Trainable parameters: {trainable_params:,}")
        print(f"   - Total parameters: {total_params:,}")
        print(f"   - Trainable %: {100 * trainable_params / total_params:.2f}%")
        
    def load_your_specific_datasets(self):
        """Load and prepare your specific datasets for COBOL to Python conversion"""
        print("Loading your specific datasets...")
        
        # Load your datasets you can change the %
        print("Loading MainframeBench COBOL dataset...")
        mainframe = load_dataset("Fsoft-AIC/MainframeBench", "COBOL_code_summarization", split="train")

        # the-stack dataset more than 3 TB of data
        print("Loading COBOL from The Stack...")
        stack_cobol = load_dataset("bigcode/the-stack", data_dir="data/cobol", split="train[:20%]")
        
        print("Loading Python datasets...")
        stack_python = load_dataset("bigcode/the-stack", data_dir="data/python", split="train[:20%]")
        python_set = load_dataset("jtatman/python-code-dataset-500k", split="train")
        
        # Combine Python datasets
        python_combined = concatenate_datasets([stack_python, python_set])
        
        # Prepare datasets for training
        all_datasets = []
        
        # 1. MainframeBench - COBOL code summarization (convert to instruction format)
        mainframe_formatted = self.format_mainframe_dataset(mainframe)
        all_datasets.append(mainframe_formatted)
        
        # 2. COBOL understanding dataset (from The Stack)
        cobol_formatted = self.format_cobol_understanding_dataset(stack_cobol)
        all_datasets.append(cobol_formatted)
        
        # 3. Python generation dataset (teach Python syntax)
        python_formatted = self.format_python_teaching_dataset(python_combined)
        all_datasets.append(python_formatted)
        
        # Combine all datasets
        self.combined_dataset = concatenate_datasets(all_datasets)
        
        # Filter by length and sample if needed
        max_length = self.config.get("max_length", 2048)
        self.combined_dataset = self.combined_dataset.filter(
            lambda x: len(self.tokenizer.encode(x["text"])) <= max_length
        )
        
        if self.config.get("max_samples"):
            self.combined_dataset = self.combined_dataset.select(
                range(min(len(self.combined_dataset), self.config["max_samples"]))
            )
            
        print(f"Combined dataset prepared: {len(self.combined_dataset)} samples")
        
    def format_mainframe_dataset(self, dataset):
        """Format MainframeBench dataset for COBOL understanding"""
        def format_mainframe(examples):
            texts = []
            for code, summary in zip(examples['code'], examples['summary']):
                prompt = f"""### Task: Analyze and explain the following COBOL code

### COBOL Code:
```cobol
{code.strip()}
```

### Explanation:
{summary.strip()}

### Convert to Python equivalent:
```python
# Python equivalent would be:
# This COBOL code performs: {summary.strip()}
```"""
                texts.append(prompt)
            return {"text": texts}
            
        return dataset.map(format_mainframe, batched=True, remove_columns=dataset.column_names)
    
    def format_cobol_understanding_dataset(self, dataset):
        """Format COBOL dataset from The Stack for understanding"""
        def format_cobol(examples):
            texts = []
            for content in examples['content']:
                # Skip very short or very long files
                if len(content.strip()) < 100 or len(content.strip()) > 5000:
                    continue
                
                prompt = f"""### Task: Understand this COBOL code and suggest Python equivalent structure

### COBOL Code:
```cobol
{content.strip()}
```

### Analysis:
This COBOL program demonstrates typical mainframe programming patterns. 

### Python Structure:
```python
# Python equivalent structure would involve:
# - Converting COBOL divisions to Python modules/classes
# - Replacing COBOL data structures with Python equivalents
# - Converting COBOL procedures to Python functions
```"""
                texts.append(prompt)
            return {"text": texts}
            
        formatted = dataset.map(format_cobol, batched=True, remove_columns=dataset.column_names)
        # Take a subset to avoid overwhelming the model
        return formatted.select(range(min(10000, len(formatted))))
    
    def format_python_teaching_dataset(self, dataset):
        """Format Python dataset to teach Python syntax and patterns"""
        def format_python(examples):
            texts = []
            for content in examples['content']:
                # Skip very short or very long files
                if len(content.strip()) < 50 or len(content.strip()) > 3000:
                    continue
                
                # Focus on clean, well-structured Python code
                if any(keyword in content.lower() for keyword in ['class ', 'def ', 'import ', 'for ', 'if ']):
                    prompt = f"""### Task: Learn Python programming patterns

### Python Code:
```python
{content.strip()}
```

### Explanation:
This Python code demonstrates modern programming practices that can be used when converting from COBOL."""
                    texts.append(prompt)
            return {"text": texts}
            
        formatted = dataset.map(format_python, batched=True, remove_columns=dataset.column_names)
        # Take a subset focused on quality code
        return formatted.select(range(min(15000, len(formatted))))
        
    def setup_training_arguments(self):
        """Setup optimized training arguments for RTX 4090"""
        return TrainingArguments(
            output_dir=self.config.get("output_dir", "./" + output_model),
            
            # Training hyperparameters optimized for code generation
            num_train_epochs=self.config.get("num_epochs", 3),
            per_device_train_batch_size=self.config.get("batch_size", 2),  # Reduced for code tasks
            gradient_accumulation_steps=self.config.get("gradient_accumulation_steps", 8),  # Increased
            learning_rate=self.config.get("learning_rate", 1e-4),  # Lower for code tasks
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            weight_decay=0.01,
            
            # NVIDIA RTX 4090 optimizations
            bf16=True,  # Use bfloat16 for Ada Lovelace
            tf32=True,  # Enable TensorFloat-32
            dataloader_pin_memory=True,
            dataloader_num_workers=4,
            
            # Memory optimizations for long code sequences
            gradient_checkpointing=True,
            optim="adamw_torch_fused",  # Fused optimizer for NVIDIA GPUs
            max_grad_norm=1.0,
            
            # Logging and saving
            logging_steps=25,
            save_steps=500,
            save_total_limit=2,
            evaluation_strategy="no",  # Disable eval to save memory
            
            # Additional optimizations
            remove_unused_columns=False,
            report_to="wandb" if self.config.get("use_wandb", False) else None,
            run_name=f"codellama-cobol-python-{self.config.get('experiment_name', 'default')}",
            
            # Prevent OOM with long sequences
            dataloader_drop_last=True,
            
            # DDP settings (single GPU for now)
            ddp_find_unused_parameters=False,
        )
        
    def train(self):
        """Execute the fine-tuning process"""
        print("🚀 Starting CodeLlama COBOL→Python fine-tuning...")
        
        # Initialize wandb if enabled
        if self.config.get("use_wandb", False):
            wandb.init(
                project=self.config.get("wandb_project", "codellama-cobol-python"),
                name=f"codellama-{self.config.get('experiment_name', 'default')}"
            )
            
        # Setup training arguments
        training_args = self.setup_training_arguments()
        
        # Data collator for code generation
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8,  # Optimize for tensor cores
        )
        
        # Initialize SFTTrainer
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.combined_dataset,
            data_collator=data_collator,
            args=training_args,
            tokenizer=self.tokenizer,
            dataset_text_field="text",
            max_seq_length=self.config.get("max_length", 2048),
            packing=False,  # Don't pack for code conversion tasks
        )
        
        # Clear cache before training
        torch.cuda.empty_cache()
        gc.collect()
        
        # Monitor initial GPU usage
        self.monitor_gpu_usage()
        
        # Start training
        print("🎯 Training started...")
        trainer.train()
        
        # Save the final model
        print("💾 Saving model...")
        trainer.save_model()
        self.tokenizer.save_pretrained(training_args.output_dir)
        
        print("✅ CodeLlama COBOL→Python fine-tuning completed!")
        
    def evaluate_model(self, test_samples=None):
        """Evaluate the fine-tuned model on COBOL to Python conversion"""
        if test_samples is None:
            # Use a small subset for quick evaluation
            test_samples = self.combined_dataset.select(range(min(10, len(self.combined_dataset))))
            
        print("🔍 Evaluating model performance...")
        
        self.model.eval()
        results = []
        
        with torch.no_grad():
            for sample in test_samples:
                # Extract COBOL code from the sample
                text = sample['text']
                cobol_start = text.find('```cobol') + 8
                cobol_end = text.find('```', cobol_start)
                cobol_code = text[cobol_start:cobol_end].strip()
                
                # Generate Python code
                prompt = f"""### Task: Convert the following COBOL code to Python

### COBOL Code:
```cobol
{cobol_code}
```

### Python Code:
```python"""
                
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                python_code = generated.split('```python')[-1].split('```')[0].strip()
                
                results.append({
                    'cobol_code': cobol_code,
                    'generated_python': python_code
                })
                
        print(f"✅ Evaluated {len(results)} samples")
        return results
        
    def monitor_gpu_usage(self):
        """Monitor GPU usage during training"""
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
                total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"GPU {i}: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved, {total_memory:.2f}GB total")

    def _extract_true_python_code(self, sample_text):
        """
        Helper to extract the true Python code from the 'text' field of a dataset sample.
        This needs to be robust to the different formatting styles you used
        across the combined datasets.
        """
        # This is the same helper function as in Option 1
        python_code_start_marker_1 = "### Python Code:\n```python"
        python_code_start_marker_2 = "### Python Structure:\n```python"
        python_code_start_marker_3 = "### Convert to Python equivalent:\n```python"
        python_end_marker = "```"

        start_index = -1
        if python_code_start_marker_1 in sample_text:
            start_index = sample_text.find(python_code_start_marker_1) + len(python_code_start_marker_1)
        elif python_code_start_marker_2 in sample_text:
            start_index = sample_text.find(python_code_start_marker_2) + len(python_code_start_marker_2)
        elif python_code_start_marker_3 in sample_text:
            start_index = sample_text.find(python_code_start_marker_3) + len(python_code_start_marker_3)

        if start_index != -1:
            end_index = sample_text.find(python_end_marker, start_index)
            if end_index != -1:
                return sample_text[start_index:end_index].strip()
        return ""


    def calculate_and_print_metrics(self, test_dataset, predictions, num_examples_to_show=3):
        """
        Calculates and prints evaluation metrics for code translation.

        Args:
            test_dataset (Dataset): The Hugging Face Dataset used for testing.
            predictions (list): A list of generated Python code strings (y_pred).
        """
        if not isinstance(test_dataset, Dataset):
            raise TypeError("test_dataset must be a Hugging Face Dataset object.")

        if len(test_dataset) != len(predictions):
            raise ValueError("Mismatch in number of samples between test_dataset and predictions.")

        # Extract ground truth (y_true)
        y_true = [self._extract_true_python_code(sample["text"]) for sample in test_dataset]

        # Filter out samples where true_code could not be extracted
        valid_pairs = [(gt, pred) for gt, pred in zip(y_true, predictions) if gt]

        if not valid_pairs:
            print("❗ No valid ground truth Python code found in the test dataset for evaluation metrics.")
            return {}

        y_true_filtered = [pair[0] for pair in valid_pairs]
        y_pred_filtered = [pair[1] for pair in valid_pairs]

        print(f"Evaluating metrics on {len(y_true_filtered)} valid samples.")

        def code_similarity(a, b):
            return SequenceMatcher(None, a.strip(), b.strip()).ratio()

        similarities = [code_similarity(gt, pred) for gt, pred in zip(y_true_filtered, y_pred_filtered)]
        avg_similarity = np.mean(similarities)

        exact_matches = sum(1 for gt, pred in zip(y_true_filtered, y_pred_filtered) if gt.strip() == pred.strip())
        accuracy = exact_matches / len(y_true_filtered)

        print(f"\n--- Evaluation Metrics ---")
        print(f"Exact Match Accuracy: {accuracy:.3f}")
        print(f"Average Similarity Score: {avg_similarity:.3f}")

        print(f"\n--- Sample Outputs ({min(num_examples_to_show, len(y_true_filtered))} examples) ---")
        for i in range(min(num_examples_to_show, len(y_true_filtered))):
            print(f"\n--- Sample {i+1} ---")
            print("True Output:\n", y_true_filtered[i])
            print("Predicted Output:\n", y_pred_filtered[i])
            print("Similarity Score:", code_similarity(y_true_filtered[i], y_pred_filtered[i]))

        return {
            "exact_match_accuracy": accuracy,
            "average_similarity": avg_similarity
        }

import numpy as np
import pandas as pd
import os
import gc
import json
import warnings
from tqdm import tqdm
from typing import Dict, List, Optional, Union

# Core ML libraries
import torch
import torch.nn as nn
import transformers
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    TrainingArguments, 
    pipeline, 
    logging,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    # CodeLlama specific
    CodeLlamaTokenizer,
    LlamaForCausalLM
)

# Dataset handling
from datasets import Dataset, load_dataset, concatenate_datasets
from sklearn.model_selection import train_test_split

# Fine-tuning libraries
import bitsandbytes as bnb
from peft import (
    LoraConfig, 
    PeftConfig, 
    get_peft_model, 
    TaskType,
    prepare_model_for_kbit_training
)
from trl import SFTTrainer, setup_chat_format

# Evaluation metrics
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix
)
from evaluate import load as load_metric

# Monitoring and optimization
import wandb
import pynvml
from accelerate import Accelerator

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
transformers.logging.set_verbosity_error()

# Set random seeds for reproducibility
def set_random_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_random_seeds(42)

print("📦 All libraries imported successfully!")
print(f"🔥 PyTorch version: {torch.__version__}")
print(f"🤗 Transformers version: {transformers.__version__}")
print(f"🚀 CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"💾 GPU: {torch.cuda.get_device_name(0)}")
    print(f"🎯 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

import os
from huggingface_hub import login
from huggingface_hub import CommitInfo

# Set environment variables in your shell or .bashrc/.zshrc
# export HUGGING_FACE_WRITE_API_KEY="hf_your_write_token_here"
# export HUGGINGFACE_TOKEN="hf_your_read_token_here"

# In your Python code
secret_value_0 = os.getenv("HUGGING_FACE_WRITE_API_KEY")
secret_value_1 = os.getenv("HUGGINGFACE_TOKEN")

if not secret_value_0 or not secret_value_1:
    raise ValueError("Please set HUGGING_FACE_WRITE_API_KEY and HUGGINGFACE_TOKEN environment variables")

hf_token = os.getenv("HUGGINGFACE_TOKEN")

# Login to Hugging Face Hub
login(token=hf_token)


# If you want to load a fresh pipeline, specify the model and tokenizer:
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id # Ensure model config also has pad_token_id


pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt = "How to convert COBOL code into Python?"
outputs = pipe(prompt, max_new_tokens=120, do_sample=True)
print(outputs[0]["generated_text"])

from datasets import DatasetDict

# Split 80% train, 10% eval, 10% test
# Use the combined_dataset from your fine_tuner instance
splits = fine_tuner.combined_dataset.train_test_split(test_size=0.2, seed=42)
eval_test = splits["test"].train_test_split(test_size=0.5, seed=42)

dataset_dict = DatasetDict({
    "train": splits["train"],
    "eval": eval_test["train"],
    "test": eval_test["test"]
})

import bitsandbytes
print(bitsandbytes.__file__)

trainer = SFTTrainer(
    model=fine_tuner.model, # Use the model with LoRA already applied by fine_tuner.setup_lora()
    args=fine_tuner.setup_training_arguments(), # Get TrainingArguments from your class
    train_dataset=dataset_dict["train"], # Use the train split from dataset_dict
    eval_dataset=dataset_dict["eval"],   # Use the eval split from dataset_dict (optional but good for validation)
    tokenizer=fine_tuner.tokenizer, # Use the tokenizer from your class
    dataset_text_field=fine_tuner.config.get("dataset_text_field", "text"), # Get from config
    max_seq_length=fine_tuner.config.get("max_length", 2048), # Get from config
    packing=fine_tuner.config.get("packing", False), # Get from config
)

print("\nTrainer successfully configured!")

# To start the fine-tuning process:
trainer.train()

# Save trained model and tokenizer

# Get the output directory path directly from the trainer's arguments
# This ensures consistency with where the trainer has been saving checkpoints
final_model_output_path = trainer.args.output_dir

trainer.save_model(final_model_output_path)
fine_tuner.tokenizer.save_pretrained(final_model_output_path)

print(f"Model and tokenizer saved to: {final_model_output_path}")

# Assuming 'fine_tuner' is your OptimizedCodeLlamaFineTuner instance
# and 'dataset_dict' contains your "test" split.

print("Starting model evaluation...")

# 1. Generate predictions using the evaluate_model method
# This method generates predictions and extracts the true labels
# The method will return a list of dictionaries, where each dict might contain
# 'cobol_code', 'generated_python' (prediction), and potentially 'true_python_code' (ground truth).
evaluation_results = fine_tuner.evaluate_model(test_dataset=dataset_dict["test"])

# From the evaluation_results, extract just the generated Python code (y_pred)
# and the true Python code (y_true) for passing to the metrics calculator.
# Note: The 'evaluate_model' method should already be structured to return these.
# If 'evaluate_model' returns predictions directly, then you extract y_pred from there.
# And y_true is extracted by calculate_and_print_metrics internally using test_dataset.

# Example: If evaluation_results is a list of dictionaries with a 'generated_python' key
y_pred_for_metrics = [item['generated_python'] for item in evaluation_results]

# 2. Calculate and print metrics using the calculate_and_print_metrics method
# This method takes the test_dataset (to re-extract y_true internally) and the predictions.
fine_tuner.calculate_and_print_metrics(test_dataset=dataset_dict["test"], predictions=y_pred_for_metrics)

print("\nEvaluation complete!")

# Assuming 'fine_tuner' is your OptimizedCodeLlamaFineTuner instance
# and the training has completed and model saved.

# The base model name is stored in your fine_tuner's config
base_model_name = fine_tuner.config["model_name"]

# The output directory for the fine-tuned model is stored in the trainer's args
# which came from your fine_tuner's config.
fine_tuned_model_path = trainer.args.output_dir

print(f"Base Model: {base_model_name}")
print(f"Fine-tuned Model Saved At: {fine_tuned_model_path}")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Use the 'base_model_name' variable defined in the previous step
# (which came from fine_tuner.config["model_name"])
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

base_model_reload = AutoModelForCausalLM.from_pretrained(
    base_model_name, # Use the defined base_model_name
    return_dict=True,
    low_cpu_mem_usage=True,
    # Use torch.bfloat16 for consistency with your config_dict's bnb_4bit_compute_dtype
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

print(f"Base model '{base_model_name}' and its tokenizer reloaded successfully.")

from peft import PeftModel

# Merge adapter with base model
# 'base_model_reload' is the AutoModelForCausalLM you just reloaded.
# 'fine_tuned_model_path' is the path where your LoRA adapters were saved by the trainer.
model = PeftModel.from_pretrained(base_model_reload, fine_tuned_model_path)

# This merges the LoRA adapter weights into the base model weights
# and removes the adapter layers, leaving you with a single, merged model.
model = model.merge_and_unload()

print(f"LoRA adapters successfully merged into the base model. The merged model is ready.")

from transformers import pipeline # Ensure pipeline is imported

cobol_code = """
        IDENTIFICATION DIVISION.
        PROGRAM-ID. HELLO.
        PROCEDURE DIVISION.
            DISPLAY 'HELLO, WORLD'.
            STOP RUN.
"""

prompt = f"""
### Instruction:
Convert the following COBOL code to Python:

{cobol_code}

### Response:
""".strip()

pipe = pipeline(
    "text-generation",
    model=model,        # This is your merged model
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16, # Use bfloat16 for consistency with training and loading
    device_map="auto"
)

outputs = pipe(prompt, max_new_tokens=256, do_sample=False, temperature=0.1)
generated_code = outputs[0]["generated_text"].split("### Response:")[-1].strip()

print("Generated Python Code:\n")
print(generated_code)

# Define the directory where you want to save your final merged model
model_dir = "CodeLlama-7B-Instruct-COBOL-to-Python"

# Save the merged model (which now contains the LoRA weights)
model.save_pretrained(model_dir)

# Save the tokenizer
tokenizer.save_pretrained(model_dir)

print(f"Your final merged model and tokenizer have been saved to: {model_dir}")

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
import torch

# Assuming base_model_name and fine_tuned_model_path are defined from previous steps:
# For example:
# base_model_name = fine_tuner.config["model_name"]
# fine_tuned_model_path = trainer.args.output_dir # or your chosen output directory for adapters

print(f"Loading tokenizer from: {base_model_name}")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

print(f"Loading base model from: {base_model_name} with torch_dtype=torch.bfloat16")
base_model_reload = AutoModelForCausalLM.from_pretrained(
    base_model_name, # Use the correct variable name for the base model identifier
    return_dict=True,
    low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16, # Use bfloat16 for consistency
    device_map="auto",
    trust_remote_code=True,
)

print(f"Loading fine-tuned adapters from: {fine_tuned_model_path}")
model = PeftModel.from_pretrained(base_model_reload, fine_tuned_model_path) # Use the correct path for adapters

print("Merging adapters into the base model...")
model = model.merge_and_unload()

print("Model and adapters merged successfully!")

!huggingface-cli login

# Save and register to Hugging Face Hub
# Ensure you are logged in to Hugging Face locally (huggingface-cli login)

# Push the merged model to your Hugging Face repository
model.push_to_hub("dhirajpatra/codellama-cobol-python", use_temp_dir=False)

# Push the tokenizer to the same repository
tokenizer.push_to_hub("dhirajpatra/codellama-cobol-python", use_temp_dir=False)

print("\nModel and tokenizer successfully pushed to Hugging Face Hub!")
print("You can find them at: https://huggingface.co/dhirajpatra/codellama-cobol-python")

from transformers import AutoModelForCausalLM, AutoTokenizer

# Assuming 'model_dir' is the string "codellama-cobol-python" you used when saving
# Or, even better, just use the full repo ID directly for clarity:
hf_repo_id = "dhirajpatra/codellama-cobol-python"

model = AutoModelForCausalLM.from_pretrained(hf_repo_id)
tokenizer = AutoTokenizer.from_pretrained(hf_repo_id)

print(f"Model and tokenizer loaded successfully from Hugging Face Hub: {hf_repo_id}")

from transformers import AutoTokenizer, AutoModelForCausalLM

# Use the full repository ID directly for clarity and robustness
# Assuming model_dir was "codellama-cobol-python" from your save step
model_id = "dhirajpatra/codellama-cobol-python"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load the model
model = AutoModelForCausalLM.from_pretrained(model_id)

print(f"**Model and tokenizer loaded successfully from Hugging Face Hub:** `{model_id}`")

from transformers import pipeline # Ensure pipeline is imported
import torch # Ensure torch is imported if not already

cobol_code = """
        IDENTIFICATION DIVISION.
        PROGRAM-ID. HELLO.
        PROCEDURE DIVISION.
            DISPLAY 'HELLO, WORLD'.
            STOP RUN.
"""

prompt = f"""### Instruction:
Convert the following COBOL code to Python:

{cobol_code}

### Response:
""".strip()

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16, # Changed to bfloat16 for consistency and performance
    device_map="auto",
)

outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.1)
generated_code = outputs[0]["generated_text"].split("### Response:")[-1].strip()

print("Generated Python Code:\n")
print(generated_code)

from datasets import load_metric, DatasetDict
# Assuming fine_tuner is your OptimizedCodeLlamaFineTuner instance
# and dataset_dict contains your "test" split.
# And assuming you've already run the evaluation_results step:
# evaluation_results = fine_tuner.evaluate_model(test_dataset=dataset_dict["test"])
# y_pred_for_metrics = [item['generated_python'] for item in evaluation_results]

# 1. Load the BLEU metric
bleu = load_metric("bleu")

# 2. Prepare predictions (preds) - This is your generated code
# Assuming y_pred_for_metrics holds the list of generated Python code strings
preds = y_pred_for_metrics

# 3. Prepare references (refs) - This requires extracting the true Python code
# The calculate_and_print_metrics method has a helper for this.
# We need to call that helper or re-extract here.
# Let's re-extract the filtered ground truth for consistency with metrics.
y_true = [fine_tuner._extract_true_python_code(sample["text"]) for sample in dataset_dict["test"]]
valid_pairs = [(gt, pred) for gt, pred in zip(y_true, preds) if gt]

# preds should be from the filtered valid_pairs as well for direct comparison
# If you ran fine_tuner.calculate_and_print_metrics, it already filtered.
# For BLEU, ensure preds and refs correspond to the same filtered set.
preds_filtered_for_bleu = [pair[1] for pair in valid_pairs]
refs_filtered_for_bleu = [[pair[0]] for pair in valid_pairs] # BLEU expects a list of lists for references

if not preds_filtered_for_bleu:
    print("❗ No valid samples to compute BLEU score on.")
else:
    # 4. Compute the BLEU score
    results = bleu.compute(predictions=preds_filtered_for_bleu, references=refs_filtered_for_bleu)

    print("\n--- BLEU Score ---")
    print(f"BLEU score: {results['bleu']:.4f}")
    # You can also print other details if available in results, e.g., 'precisions'
    # print(results)
# LoRA_WandB_Gemma2_2b_en
This repository provides a Colab-ready Python script for fine-tuning a GemmaCausalLM model using LoRA with KerasNLP. It covers Colab setup, downloading and preprocessing the Dolly 15k dataset, training with different LoRA ranks, and integrating Weights &amp; Biases for experiment tracking.


Overview
-Objective: Fine-tune a Gemma model (2B parameters) with LoRA to improve text generation performance on specific tasks.
-Key Features:
  -Mounting Google Drive to store and load model weights.
  -Environment variable configuration for Kaggle API credentials.
  -Downloading and preprocessing the Databricks Dolly 15k dataset.
  -Fine-tuning with different LoRA ranks (4, 8, 16, 32) and mixed precision training.
  -Integration with Weights & Biases for experiment tracking and logging.

Requirements
-Python: 3.7+
-Dependencies:
  -Keras
  -KerasNLP
  -Weights & Biases (wandb)
  -Other libraries as specified in requirements.txt
  Make sure to install these packages either via pip or by running the installation commands in the Colab notebook.

Setup in Colab
1.Mount Google Drive: The script mounts Google Drive to load/save model weights.
2.Set Environment Variables: Environment variables for Kaggle credentials are set for downloading datasets.
3.Backend Selection: The script configures the Keras backend (e.g., JAX, TensorFlow, or PyTorch). For this tutorial, JAX is used with settings to avoid memory fragmentation.
4.Install Dependencies: Ensure that Keras and KerasNLP are updated to the correct versions.

Running the Notebook
1.Open the Colab notebook or run the lora_gemma2_2b_en.py script in a Colab environment.
2.Execute each code cell sequentially to:
  -Mount the drive.
  -Download and preprocess the dataset.
  -Instantiate and fine-tune the Gemma model with LoRA.
  -Save the trained model weights back to Google Drive.
3.Adjust parameters like the LoRA rank, batch size, or epochs as needed for your experiments.

Weights & Biases Integration
To track experiments with Weights & Biases, follow these steps:

1.Install wandb:
<pre lang="no-highlight"> ```
!pip install wandb 
```</pre>

2.Initialize wandb in your script:
At the beginning of the script (after imports), add:
<pre lang="no-highlight"> ```
import wandb
wandb.init(project="gemma-lora-finetuning", entity="your_wandb_username")
``` </pre>

3.Log Hyperparameters and Metrics:
During model training, log the loss and accuracy using a custom callback or by inserting logging calls in your training loop. For example:
<pre lang="no-highlight"> ```
class WandbCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        wandb.log(logs)
 ``` </pre>

# Then include the callback in your model.fit call:
<pre lang="no-highlight"> ```
gemma_lm.fit(data, epochs=1, batch_size=1, callbacks=[WandbCallback()])
``` </pre>

4.Monitor Training:
Track model performance in real time through your Weights & Biases dashboard.

Project Structure
  project/
  ├── README.md                # Project overview, usage instructions, and background information (this file)
  ├── .gitignore               # Specifies files and directories to be ignored by Git
  ├── requirements.txt         # List of required Python packages (keras, keras-nlp, wandb, etc.)
  ├── lora_gemma2_2b_en.py       # Main Python script for fine-tuning the Gemma model
  ├── data/                    # Data directory
  │   └── databricks-dolly-15k.jsonl  # Dataset file downloaded during execution
  ├── models/                  # Directory to store model weights
  │   ├── my_gemma_lm_weights.weights.h5
  │   ├── my_gemma_lm_weights_mixed_precision.weights.h5
  │   ├── my_gemma_lm_weights_mixed_precision_rank8.weights.h5
  │   ├── my_gemma_lm_weights_mixed_precision_rank16.weights.h5
  │   └── my_gemma_lm_weights_mixed_precision_rank32.weights.h5
  └── notebooks/               # (Optional) Original Colab notebook files
      └── LoRA_gemma2_2b_en.ipynb

Additional Information
Documentation & References:
[KerasNLP Documentation](https://keras.io/keras_hub/api/)

[LoRA Tuning Example](https://keras.io/examples/nlp/parameter_efficient_finetuning_of_gpt2_with_lora/)

[Weights & Biases Documentation](https://docs.wandb.ai/guides/track/launch/)

[Gemma Model Documentation](https://ai.google.dev/gemma/docs/get_started)

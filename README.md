# Fine-Tuning DistilBERT for Masked Language Modeling (MLM)

## Overview

This notebook demonstrates how to fine-tune DistilBERT for the Masked Language Modeling (MLM) task. The goal is to train the model to predict masked tokens in a sequence, which is a fundamental pre-training task for transformer-based language models. The notebook uses the IMDB dataset for training and evaluation, providing a practical example of how to adapt a pre-trained model for MLM.

https://github.com/elkomy13/Fine-Tune-DistilBERT-For-Masked-Tasks/mask.mp4

## Key Features

- **Model**: DistilBERT (distilbert-base-uncased), a lightweight and efficient version of BERT.
- **Task**: Masked Language Modeling (MLM), where the model predicts randomly masked tokens in a sequence.
- **Dataset**: IMDB movie reviews, used for fine-tuning and evaluation.
- **Techniques**:
  - Whole Word Masking (WWM) to mask entire words instead of subword tokens.
  - Dynamic masking during training for better generalization.
  - Perplexity evaluation to measure model performance.
- **Deployment** : StreamLit

## Workflow

1. **Setup**: Load the pre-trained DistilBERT model and tokenizer.
2. **Data Preparation**:
   - Tokenize the IMDB dataset.
   - Concatenate and chunk the tokenized data for MLM.
   - Apply dynamic masking with a 15% probability.
3. **Training**:
   - Use the `Trainer` class from Hugging Face for fine-tuning.
   - Accelerate training with mixed-precision (FP16) and gradient accumulation.
4. **Evaluation**:
   - Measure model performance using perplexity.
   - Compare results across epochs.


## Usage

1. **Install Dependencies**:
   ```bash
   pip install transformers datasets torch accelerate tqdm
   ```

2. **Run the Notebook**:
   - Execute the cells sequentially to load the model, preprocess the data, and train the model.
   - Adjust hyperparameters (e.g., `batch_size`, `learning_rate`) as needed.

3. **Customization**:
   - Replace the IMDB dataset with your own corpus for domain-specific fine-tuning.
   - Modify the masking probability (`mlm_probability`) or chunk size (`chunk_size`) for different use cases.

## Results

- The model's performance is evaluated using perplexity, with lower values indicating better performance.
- Example perplexity after 3 epochs: ~10.89.

## Notes

- **Whole Word Masking**: Ensures all subword tokens of a masked word are masked together, improving training stability.
- **Downsampling**: The notebook downsamples the dataset for faster experimentation. Remove this step for full training.

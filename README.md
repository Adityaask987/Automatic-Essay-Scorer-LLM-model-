# Automatic Essay Scorer

## Overview
The Automatic Essay Scorer is an AI-driven project that leverages a Large Language Model (LLM) to evaluate and score essays automatically. The model is fine-tuned on a dataset of student essays, providing accurate and consistent scoring, which can be used as a tool for educational purposes, especially in environments where automated grading is beneficial.

## Features
- **Model**: Utilizes the DeBERTa v3 model for understanding and evaluating essays.
- **Fine-Tuning**: The model is fine-tuned on a custom dataset of essays to achieve better accuracy in scoring.
- **Scoring Metrics**: Outputs include mean, standard deviation, and quartiles of the scores.
- **Customizable**: The model can be adapted to various essay datasets and scoring rubrics.

## Installation
To run the Automatic Essay Scorer, ensure you have the following installed:

```bash
pip install torch transformers pandas numpy scikit-learn
```
# Usage
## Model Training:

Fine-tune the DeBERTa v3 model on your specific essay dataset using the provided code in the notebook.
## Scoring Essays:

Use the trained model to score new essays by feeding them into the model and receiving a score between a predefined range.
## Evaluate Results:

The model provides detailed statistical output, including the mean, standard deviation, and quartile distribution of the essay scores.

# Example

## Example of loading the model and scoring an essay
```bash
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
model = AutoModelForSequenceClassification.from_pretrained("your-fine-tuned-model")

essay = "Your essay text here"
inputs = tokenizer(essay, return_tensors="pt", padding=True, truncation=True)
outputs = model(**inputs)
score = outputs.logits.argmax().item()

print(f"Predicted score: {score}")
```
## Dataset
The dataset used for training the model should consist of essays and their respective scores. Ensure the dataset is preprocessed and split into training and validation sets for optimal model performance.

## Contributing
Contributions to this project are welcome. If you find any issues or have suggestions, feel free to submit a pull request or open an issue.

## License
This project is licensed under the MIT License.

## Acknowledgments
Special thanks to the creators of the DeBERTa model and the educational institutions that provided the essay datasets used in training.


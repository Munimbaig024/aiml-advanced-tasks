# Task 5: Customer Support Ticket Classification with LLMs

## 📋 Overview

This project implements an intelligent customer support ticket classification system using Large Language Models (LLMs). The system automatically categorizes incoming support tickets into predefined categories to improve response times and routing efficiency.

### 🎯 Objectives
- Automatically classify customer support tickets into appropriate categories
- Compare performance of different LLM approaches (Zero-shot vs Few-shot)
- Provide comprehensive evaluation metrics and analysis
- Demonstrate practical application of LLMs in customer service automation

## 📊 Dataset Information

- **Dataset**: `customer_support_tickets.csv`
- **Total Tickets**: 8,469
- **Ticket Categories**: 5 unique types
- **Features**: Ticket ID, Ticket Type, Ticket Description

### Sample Data Structure
```csv
Ticket ID,Ticket Type,Ticket Description
1,Technical issue,"I'm having an issue with the {product_purchased}. Please assist."
2,Billing inquiry,"I need help with my billing statement..."
```

## 🏗️ Architecture & Methodology

### 1. Data Preprocessing
- Load and clean customer support ticket data
- Remove duplicates and handle missing values
- Extract relevant features (Ticket Type, Ticket Description)

### 2. Model Approaches

#### A. Zero-Shot Classification (BART-large-MNLI)
- **Model**: `facebook/bart-large-mnli`
- **Approach**: Uses pre-trained model without fine-tuning
- **Advantages**: No training required, works out-of-the-box
- **Use Case**: Quick deployment and baseline performance

#### B. Few-Shot Classification (FLAN-T5)
- **Model**: `google/flan-t5-base`
- **Approach**: Uses prompt engineering with few-shot examples
- **Advantages**: Better control over classification logic
- **Use Case**: Custom classification rules and domain-specific requirements

### 3. Classification Categories
- **Technical Issue**: Product functionality problems
- **Billing Inquiry**: Payment and billing-related questions
- **Account Access**: Login and account management issues
- **Product Inquiry**: General product information requests
- **Feature Request**: New feature suggestions and enhancements

## 🚀 Setup & Installation

### Prerequisites
```bash
Python 3.8+
pip install pandas numpy scikit-learn matplotlib seaborn transformers torch
```

### Installation Steps
1. Clone the repository
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the dataset (`customer_support_tickets.csv`)
4. Run the Jupyter notebook: `Task5.ipynb`


## 🔍 Key Features

### 1. Zero-Shot Classification
```python
from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
prediction = classifier(description, candidate_labels, multi_label=False)
```

### 2. Few-Shot Classification
```python
def classify_ticket(ticket):
    prompt = f"""
    Classify the following support ticket into one of the following categories:
    - Account access
    - Billing inquiry
    - Technical issue
    
    Ticket: "{ticket}"
    Category:"""
    
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=20)
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return prediction
```

### 3. Comprehensive Evaluation
```python
# Calculate metrics
accuracy = accuracy_score(actuals, predictions)
precision, recall, f1, support = precision_recall_fscore_support(
    actuals, predictions, average='weighted'
)

# Confusion matrix
cm = confusion_matrix(actuals, predictions, labels=label_encoder.classes_)
```

## 📊 Results & Performance

### Model Performance Summary
- **Zero-Shot BART**: Baseline performance with no training
- **FLAN-T5 Few-Shot**: Improved performance with prompt engineering
- **Evaluation Sample**: 100 tickets for comprehensive analysis

### Key Insights
- Model accuracy varies by ticket category
- Technical issues are generally easier to classify
- Billing inquiries show higher misclassification rates
- Prompt engineering significantly improves performance

## 🛠️ Usage Examples

### Basic Classification
```python
# Load data
df = pd.read_csv("customer_support_tickets.csv")

# Classify a single ticket
ticket_description = "I can't log into my account"
prediction = classify_ticket(ticket_description)
print(f"Predicted Category: {prediction}")
```

### Batch Classification
```python
# Classify multiple tickets
results = []
for _, row in df.head(10).iterrows():
    prediction = classifier(row['Ticket Description'], candidate_labels)
    results.append({
        'ticket': row['Ticket Description'][:100],
        'actual': row['Ticket Type'],
        'predicted': prediction['labels'][0],
        'confidence': prediction['scores'][0]
    })
```

### Evaluation
```python
# Run comprehensive evaluation
evaluate_model(df.head(100), classifier)
```

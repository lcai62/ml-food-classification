# ML Food Survey Classification

This project explores survey responses related to three food items — **Pizza**, **Shawarma**, and **Sushi** to build a custom multi-class classification model.

---

## Dataset

Survey questions included:

- Complexity rating (1 to 5)
- Expected number of ingredients (free-form text)
- Expected settings for serving (multi-select)
- Expected price (free-form text)
- Movie association (free-form text)
- Drink pairing (free-form text)
- People association (multi-select)
- Hot sauce preference (multi-select)

Exact survey questions are available # todo

---

### Final Model Highlights

- Accuracy: ~**88.4%**
- Trained using 100 independent runs
- One-hot and BOW encoding
- Softmax activation
- L2 Regularization: 0.005
- Learning Rate: 0.005
- 2000 iterations

---

## Model Features

- **Q1**: Complexity (numeric)
- **Q2**: Parsed ingredient count (numeric)
- **Q3**: One-hot encoding of selected settings
- **Q4**: Parsed price (numeric)
- **Q5**: Bag-of-words representation of associated movies
- **Q6**: Drink category (manually curated → one-hot encoding)
- **Q7**: One-hot encoding of associated people
- **Q8**: Hot sauce preference (mapped from text to scale 0–4)

---

## Evaluation & Results

### Accuracy Results (100 Runs)

- **Training**: 91.2%
- **Validation**: 88.9%
- **Test**: 88.4%

### Figures

- Accuracy stripplot: `test_accuracy_stripplot.png`
- Heatmaps and bar charts for various features (drinks, people, movies)

---

# Getting Started

1. Clone the repo:
```bash
git clone https://github.com/lcai62/ml-food-classification/
cd ml-food-classification
```

2. Install requirements
```bash
pip install -r requirements.txt
```



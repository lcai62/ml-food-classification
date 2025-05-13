# ML Food Survey Classification

This project explores survey responses related to three food items — **Pizza**, **Shawarma**, and **Sushi** to build a custom multi-class classification model.


## Built With

[![Python][python-shield]][python-url]  
[![NumPy][numpy-shield]][numpy-url]  
[![Pandas][pandas-shield]][pandas-url]  
[![Matplotlib][matplotlib-shield]][matplotlib-url]


---

## Dataset

Survey questions included in ```questions.md```

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

3. Train model
```bash
python3 generate.py
```

4. Make predictions
```bash
python3 pred.py
```

[python-shield]: https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white
[python-url]: https://www.python.org/

[fastapi-shield]: https://img.shields.io/badge/fastapi-005571?style=for-the-badge&logo=fastapi
[fastapi-url]: https://fastapi.tiangolo.com/

[react-shield]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[react-url]: https://reactjs.org/

[electron-shield]: https://img.shields.io/badge/Electron-2C2E3B?style=for-the-badge&logo=electron&logoColor=9FEAF9
[electron-url]: https://www.electronjs.org/

[tailwind-shield]: https://img.shields.io/badge/TailwindCSS-06B6D4?style=for-the-badge&logo=tailwindcss&logoColor=white
[tailwind-url]: https://tailwindcss.com/

[numpy-shield]: https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=numpy
[numpy-url]: https://numpy.org/

[pandas-shield]: https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas
[pandas-url]: https://pandas.pydata.org/

[matplotlib-shield]: https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=matplotlib
[matplotlib-url]: https://matplotlib.org/

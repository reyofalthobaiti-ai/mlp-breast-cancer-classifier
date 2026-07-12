# MLP from Scratch — Breast Cancer Classification

A multilayer perceptron (30 → 16 → 1) implemented from scratch using only NumPy — no scikit-learn model, no PyTorch/TensorFlow. Trained to classify tumors in the Wisconsin Breast Cancer dataset as **Malignant** or **Benign**.

## What's Included

- Manual forward propagation, backpropagation, and gradient descent
- Sigmoid activation, squared error loss
- Training loop with per-epoch loss/accuracy tracking
- Comparison plots across different learning rates and hidden layer sizes
- Evaluation with accuracy, precision, recall, and F1-score

## Dataset

Automatically downloaded from the UCI Machine Learning Repository:
`https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data`

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python mlp_breast_cancer.py
```

This will:
1. Download and load the dataset
2. Split into train/test (80/20, stratified)
3. Normalize features
4. Train the MLP (30 → 16 → 1, LR=0.1, 200 epochs)
5. Save `loss_accuracy.png`, `learning_rates.png`, and `hidden_nodes.png`
6. Print evaluation metrics (accuracy, precision, recall, F1)

## Known Issues

> **Note:** This code is committed as-is, including a couple of existing bugs, for transparency:
- `initialize_weights()` has an incomplete `np.random.` call missing the function name (e.g. `randn`) — this will raise a `SyntaxError` before the script runs.
- The evaluation cell is missing the `f` in `from sklearn.metrics import ...` (also a `SyntaxError`).

Both need a one-character fix to run. See commit history / issues for details.

## License

For educational use.

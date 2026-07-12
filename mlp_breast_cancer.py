import numpy as np
import urllib.request
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score


#  SECTION 1: Activation Functions
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1.0 - a)


# Weight Initialization
def initialize_weights(n_input, n_hidden, n_output, random_state=None):

    if random_state is not None:
        np.random.seed(random_state)

    W1 = np.random.(n_input,  n_hidden) * 0.01
    b1 = np.zeros((1, n_hidden))
    W2 = np.random.rand(n_hidden, n_output) * 0.01
    b2 = np.zeros((1, n_output))

    return W1, b1, W2, b2


#  Forward Propagation

def forward_propagation(X, W1, b1, W2, b2):

    Z1 = X.dot(W1) + b1      # (n_samples x n_hidden)
    A1 = sigmoid(Z1)          # (n_samples x n_hidden)
    Z2 = A1.dot(W2) + b2     # (n_samples x n_output)
    A2 = sigmoid(Z2)          # (n_samples x n_output) — final prediction
    return Z1, A1, Z2, A2


#  Loss Function (Squared Error)

def squared_error(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1, 1)
    y_pred = np.asarray(y_pred).reshape(-1, 1)
    return np.mean(((y_pred - y_true) ** 2) / 2.0)


def squared_error_derivative(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1, 1)
    y_pred = np.asarray(y_pred).reshape(-1, 1)
    return y_pred - y_true


#   Backpropagation

def backpropagation(X, y, Z1, A1, Z2, A2, W1, W2, b1, b2):

    y = np.asarray(y).reshape(-1, 1)

    # ── Output layer gradients
    dZ2 = squared_error_derivative(y, A2) * sigmoid_derivative(A2)  # (n_samples, 1)
    dW2 = A1.T.dot(dZ2)                                              # (n_hidden,  1)
    db2 = dZ2.sum(axis=0, keepdims=True)                             # (1, 1)

    # ── Hidden layer gradients
    dA1 = dZ2.dot(W2.T)                           # (n_samples, n_hidden)
    dZ1 = dA1 * sigmoid_derivative(A1)            # (n_samples, n_hidden)
    dW1 = X.T.dot(dZ1)                            # (n_input,   n_hidden)
    db1 = dZ1.sum(axis=0, keepdims=True)          # (1, n_hidden)

    return dW1, db1, dW2, db2


#   Accuracy

def compute_accuracy(y_true, y_pred_prob, threshold=0.5):

    y_true      = np.asarray(y_true).flatten()
    y_pred_prob = np.asarray(y_pred_prob).flatten()

    correct = 0
    for i in range(len(y_true)):
        predicted = 1 if y_pred_prob[i] >= threshold else 0
        if predicted == y_true[i]:
            correct += 1

    acc = correct / float(len(y_true))
    return acc


#  Training Loop

def train(X_train, Y_train, X_test, Y_test,
          n_hidden=16, learning_rate=0.1, epochs=200, random_state=42):

    n_input  = X_train.shape[1]   # 30 features
    n_output = 1                   # binary output

    # Initialize weights
    W1, b1, W2, b2 = initialize_weights(n_input, n_hidden, n_output, random_state)

    # Lists to track progress
    train_losses,     test_losses     = [], []
    train_accuracies, test_accuracies = [], []

    for epoch in range(epochs):

        # Step 1: Forward pass on training data
        Z1, A1, Z2, A2 = forward_propagation(X_train, W1, b1, W2, b2)

        # Step 2: Compute training loss
        train_loss = squared_error(Y_train, A2)

        # Step 3: Backpropagation — compute gradients
        dW1, db1_grad, dW2, db2_grad = backpropagation(
            X_train, Y_train, Z1, A1, Z2, A2, W1, W2, b1, b2
        )

        # Step 4: Update weights using gradient descent
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1_grad
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2_grad

        # Step 5: Evaluate on test set (no weight update)
        _, _, _, A2_test = forward_propagation(X_test, W1, b1, W2, b2)
        test_loss = squared_error(Y_test, A2_test)

        # Step 6: Compute accuracy for both sets
        train_acc = compute_accuracy(Y_train, A2)
        test_acc  = compute_accuracy(Y_test,  A2_test)

        # Step 7: Save metrics for plotting
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        # Print progress every 20 epochs
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:>3}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

    return W1, b1, W2, b2, train_losses, test_losses, train_accuracies, test_accuracies


#   Plot Loss and Accuracy

def plot_results(train_losses, test_losses, train_accuracies, test_accuracies):

    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    #Loss plot
    ax1.plot(epochs, train_losses, label='Train Loss',  color='blue')
    ax1.plot(epochs, test_losses,  label='Test Loss',   color='orange')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Number of Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy plot
    ax2.plot(epochs, train_accuracies, label='Train Accuracy', color='blue')
    ax2.plot(epochs, test_accuracies,  label='Test Accuracy',  color='orange')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Number of Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('loss_accuracy.png', dpi=150)
    plt.show()
    print("Figure saved as: loss_accuracy.png")


#  Plot Different Learning Rates

def plot_learning_rates(X_train, Y_train, X_test, Y_test,
                        learning_rates=[1.0, 0.5, 0.1, 0.01],
                        epochs=100, n_hidden=16):

    colors = ['blue', 'green', 'orange', 'red']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle('Performance vs Epochs for Different Learning Rates', fontsize=13)

    for lr, color in zip(learning_rates, colors):
        print(f"  Training with LR={lr}...")
        _, _, _, _, tr_losses, ts_losses, _, _ = train(
            X_train, Y_train, X_test, Y_test,
            n_hidden=n_hidden, learning_rate=lr, epochs=epochs, random_state=42
        )
        ep = range(1, epochs + 1)
        ax1.plot(ep, tr_losses, label=f'LR={lr}', color=color)
        ax2.plot(ep, ts_losses, label=f'LR={lr}', color=color)

    ax1.set_title('Train Loss per Learning Rate')
    ax1.set_xlabel('Training Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.set_title('Test Loss per Learning Rate')
    ax2.set_xlabel('Training Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('learning_rates.png', dpi=150)
    plt.show()
    print("Figure saved as: learning_rates.png")


#  Plot Different Hidden Layer Sizes

def plot_hidden_nodes(X_train, Y_train, X_test, Y_test,
                      hidden_sizes=[5, 10, 15, 20, 25, 30],
                      epochs=100, learning_rate=0.1):

    train_accs = []
    test_accs  = []

    for k in hidden_sizes:
        print(f"  Training with hidden nodes k={k}...")
        _, _, _, _, _, _, tr_acc, ts_acc = train(
            X_train, Y_train, X_test, Y_test,
            n_hidden=k, learning_rate=learning_rate, epochs=epochs, random_state=42
        )
        train_accs.append(tr_acc[-1])   # final epoch accuracy
        test_accs.append(ts_acc[-1])

    # ── Bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    x     = np.arange(len(hidden_sizes))
    width = 0.35

    bars1 = ax.bar(x - width/2, train_accs, width, label='Train Accuracy', color='steelblue')
    bars2 = ax.bar(x + width/2, test_accs,  width, label='Test Accuracy',  color='orange')

    ax.set_title('MLP Performance vs Hidden Layer Size (Bar Chart)')
    ax.set_xlabel('Number of Hidden Nodes (k)')
    ax.set_ylabel('Classification Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(hidden_sizes)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis='y')

    # Add value labels on top of each bar
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig('hidden_nodes.png', dpi=150)
    plt.show()
    print("Figure saved as: hidden_nodes.png")

    # Summary table
    print("\nHidden Nodes Summary:")
    print(f"{'k':<6} {'Train Acc':<14} {'Test Acc'}")
    print("-" * 32)
    for k, tr, ts in zip(hidden_sizes, train_accs, test_accs):
        print(f"{k:<6} {tr:<14.4f} {ts:.4f}")


#   Dataset Loading

def download_dataset(
    url="https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
    save_path="wdbc.data"
):

    urllib.request.urlretrieve(url, save_path)
    print(f"Dataset downloaded to: {save_path}")


def load_dataset(file):

    lines = open(file).read().splitlines()
    X = np.zeros((len(lines), 30))
    Y = np.zeros(len(lines))
    print("The shape of training data is:", X.shape)

    for i, line in enumerate(lines):
        parts = line.strip().split(",")
        X[i] = parts[2:]
        Y[i] = 1 if parts[1] == 'M' else 0

    return X, Y


if __name__ == "__main__":

    #Download and load dataset
    download_dataset()
    X, Y = load_dataset("wdbc.data")

    #Split into train (80%) and test (20%) with stratification
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, stratify=Y, random_state=42
    )
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

    # Normalize features
    X_mean  = X_train.mean(axis=0)
    X_std   = X_train.std(axis=0) + 1e-8   # +1e-8 avoids division by zero
    X_train = (X_train - X_mean) / X_std
    X_test  = (X_test  - X_mean) / X_std
    print("Features normalized (zero mean, unit variance).\n")




    # ── Step 4: Train MLP (30 inputs → 16 hidden → 1 output)
    print("=" * 60)
    print("Training MLP: 30 → 16 → 1  |  LR=0.1  |  Epochs=200")
    print("=" * 60)
    W1, b1, W2, b2, train_losses, test_losses, train_acc, test_acc = train(
        X_train, Y_train, X_test, Y_test,
        n_hidden=16,
        learning_rate=0.1,
        epochs=200
    )

    # ── Step 5: Plot train/test loss and accuracy
    plot_results(train_losses, test_losses, train_acc, test_acc)

    # ── Step 6: Compare different learning rates
    print("\n" + "=" * 60)
    print("Comparing Learning Rates: [1.0, 0.5, 0.1, 0.01]")
    print("=" * 60)
    plot_learning_rates(X_train, Y_train, X_test, Y_test,
                        learning_rates=[1.0, 0.5, 0.1, 0.01],
                        epochs=100, n_hidden=16)

    # ── Step 7: Compare different hidden layer sizes
    print("\n" + "=" * 60)
    print("Comparing Hidden Layer Sizes: [5, 10, 15, 20, 25, 30]")
    print("=" * 60)
    plot_hidden_nodes(X_train, Y_train, X_test, Y_test,
                      hidden_sizes=[5, 10, 15, 20, 25, 30],
                      epochs=100, learning_rate=0.1)


rom sklearn.metrics import precision_score, recall_score, f1_score

_, _, _, A2_final = forward_propagation(X_test, W1, b1, W2, b2)

y_pred = (A2_final.flatten() >= 0.5).astype(int)
y_true = Y_test.astype(int)

accuracy  = compute_accuracy(y_true, A2_final)
precision = precision_score(y_true, y_pred)
recall    = recall_score(y_true, y_pred)
f1        = f1_score(y_true, y_pred)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")

{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNLVq5dxuQg4lTEVMfeYGuT",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/sandip988/AI-Lab2-Sandip-giri-021-348/blob/main/kNN%20for%20Multi-Class%20Classification%20with%20Arbitrary%20Features.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 2,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "bkrq49Ap8Thr",
        "outputId": "3b87a381-87ee-4d14-c854-f99121cb025e"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Accuracy: 66.67%\n",
            "Predicted class for new point [6.  3.  4.5]: Species C\n",
            "\n",
            "Test set predictions:\n",
            "Point 1: [5.8 2.7 5.1], True class: Species C, Predicted class: Species C\n",
            "Point 2: [6.3 3.3 6. ], True class: Species C, Predicted class: Species B\n",
            "Point 3: [5.1 3.5 1.4], True class: Species A, Predicted class: Species A\n"
          ]
        }
      ],
      "source": [
        "import numpy as np\n",
        "from collections import Counter\n",
        "from sklearn.model_selection import train_test_split\n",
        "\n",
        "# Function to calculate Euclidean distance\n",
        "def euclidean_distance(point1, point2):\n",
        "    return np.sqrt(np.sum((point1 - point2) ** 2))\n",
        "\n",
        "# kNN algorithm implementation for arbitrary number of features\n",
        "def knn_predict(X_train, y_train, X_test, k):\n",
        "    predictions = []\n",
        "\n",
        "    for test_point in X_test:\n",
        "        # Calculate distances between test point and all training points\n",
        "        distances = [euclidean_distance(test_point, x) for x in X_train]\n",
        "\n",
        "        # Get indices of k nearest neighbors\n",
        "        k_indices = np.argsort(distances)[:k]\n",
        "\n",
        "        # Get the labels of the k nearest neighbors\n",
        "        k_nearest_labels = [y_train[i] for i in k_indices]\n",
        "\n",
        "        # Predict the class based on majority vote\n",
        "        most_common = Counter(k_nearest_labels).most_common(1)[0][0]\n",
        "        predictions.append(most_common)\n",
        "\n",
        "    return np.array(predictions)\n",
        "\n",
        "# Function to calculate accuracy\n",
        "def calculate_accuracy(y_true, y_pred):\n",
        "    return np.mean(y_true == y_pred)\n",
        "\n",
        "# Load and prepare the dataset\n",
        "data = np.array([\n",
        "    [5.1, 3.5, 1.4, 0],  # Species A\n",
        "    [4.9, 3.0, 1.3, 0],  # Species A\n",
        "    [5.0, 3.4, 1.5, 0],  # Species A\n",
        "    [7.0, 3.2, 4.7, 1],  # Species B\n",
        "    [6.4, 3.2, 4.5, 1],  # Species B\n",
        "    [6.9, 3.1, 4.9, 1],  # Species B\n",
        "    [5.5, 2.3, 4.0, 2],  # Species C\n",
        "    [6.5, 2.8, 4.6, 2],  # Species C\n",
        "    [5.7, 2.8, 4.1, 2],  # Species C\n",
        "    [6.3, 3.3, 6.0, 2],  # Species C\n",
        "    [5.8, 2.7, 5.1, 2],  # Species C\n",
        "    [6.1, 3.0, 4.8, 2]   # Species C\n",
        "])\n",
        "\n",
        "# Split features and labels\n",
        "X = data[:, :-1]  # All features (sepal length, sepal width, petal length)\n",
        "y = data[:, -1].astype(int)  # Class labels (0, 1, 2 for Species A, B, C)\n",
        "\n",
        "# Split the dataset into training and testing sets (80-20 split)\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
        "\n",
        "# Set k value\n",
        "k = 5\n",
        "\n",
        "# Make predictions\n",
        "y_pred = knn_predict(X_train, y_train, X_test, k)\n",
        "\n",
        "# Calculate and print accuracy\n",
        "accuracy = calculate_accuracy(y_test, y_pred)\n",
        "print(f\"Accuracy: {accuracy * 100:.2f}%\")\n",
        "\n",
        "# Example prediction for a new test point\n",
        "new_point = np.array([6.0, 3.0, 4.5])\n",
        "prediction = knn_predict(X_train, y_train, [new_point], k)\n",
        "class_names = {0: 'Species A', 1: 'Species B', 2: 'Species C'}\n",
        "print(f\"Predicted class for new point {new_point}: {class_names[prediction[0]]}\")\n",
        "\n",
        "# Print test set predictions\n",
        "print(\"\\nTest set predictions:\")\n",
        "for i, (point, true_label, pred_label) in enumerate(zip(X_test, y_test, y_pred)):\n",
        "    print(f\"Point {i+1}: {point}, True class: {class_names[true_label]}, \"\n",
        "          f\"Predicted class: {class_names[pred_label]}\")"
      ]
    }
  ]
}
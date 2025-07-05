{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNZ+MF8Wehphb+k3O6FLTMl",
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
        "<a href=\"https://colab.research.google.com/github/sandip988/AI-Lab2-Sandip-giri-021-348/blob/main/kNN%20for%20Binary%20Classification%20with%202%20Features.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "g9rRW_fw7lah",
        "outputId": "1350863e-eae6-4fc7-c994-11fe79d775d1"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Accuracy: 100.00%\n",
            "Predicted class for new point [166  63]: Pass\n",
            "\n",
            "Test set predictions:\n",
            "Point 1: [158  53], True class: Fail, Predicted class: Fail\n",
            "Point 2: [170  65], True class: Pass, Predicted class: Pass\n"
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
        "# kNN algorithm implementation\n",
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
        "    [165, 60, 1],\n",
        "    [170, 65, 1],\n",
        "    [160, 55, 0],\n",
        "    [175, 70, 1],\n",
        "    [155, 50, 0],\n",
        "    [168, 62, 1],\n",
        "    [162, 58, 0],\n",
        "    [172, 68, 1],\n",
        "    [158, 53, 0],\n",
        "    [167, 61, 1]\n",
        "])\n",
        "\n",
        "# Split features and labels\n",
        "X = data[:, :2]  # Height and weight\n",
        "y = data[:, 2].astype(int)  # Class labels\n",
        "\n",
        "# Split the dataset into training and testing sets (80-20 split)\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
        "\n",
        "# Set k value\n",
        "k = 3\n",
        "\n",
        "# Make predictions\n",
        "y_pred = knn_predict(X_train, y_train, X_test, k)\n",
        "\n",
        "# Calculate and print accuracy\n",
        "accuracy = calculate_accuracy(y_test, y_pred)\n",
        "print(f\"Accuracy: {accuracy * 100:.2f}%\")\n",
        "\n",
        "# Example prediction for a new test point\n",
        "new_point = np.array([166, 63])\n",
        "prediction = knn_predict(X_train, y_train, [new_point], k)\n",
        "print(f\"Predicted class for new point {new_point}: {'Pass' if prediction[0] == 1 else 'Fail'}\")\n",
        "\n",
        "# Print test set predictions\n",
        "print(\"\\nTest set predictions:\")\n",
        "for i, (point, true_label, pred_label) in enumerate(zip(X_test, y_test, y_pred)):\n",
        "    print(f\"Point {i+1}: {point}, True class: {'Pass' if true_label == 1 else 'Fail'}, \"\n",
        "          f\"Predicted class: {'Pass' if pred_label == 1 else 'Fail'}\")"
      ]
    }
  ]
}
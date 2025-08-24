# Blood Donor Availability Predictor 🩺💉

Welcome to the **Blood Donor Availability Predictor** project! This
repository contains a deep learning model built with TensorFlow to
predict whether blood donors are available to donate based on their
donation history and blood group. 🚀

## 📖 Project Overview

This project uses a neural network to classify blood donors as available
(`Yes`) or unavailable (`No`) for donation. The dataset includes
features like months since first donation, number of donations, pints
donated, and blood group. The model is trained with advanced techniques
like batch normalization, dropout, and learning rate scheduling to
achieve high accuracy. 📊

### Key Features

-   **Data Preprocessing**: Handles categorical data (blood group) with
    label encoding and scales numerical features. 🛠️
-   **Neural Network**: A deep learning model with multiple layers,
    batch normalization, and dropout for robust predictions. 🧠
-   **Evaluation**: Includes accuracy metrics and ROC curve
    visualization for model performance. 📈
-   **Visualization**: Plots ROC curves to assess the model's
    discriminatory power. 🎨

## 🛠️ Installation

To run this project, ensure you have Python 3.8+ installed. Follow these
steps:

1.  Clone the repository:

    ``` bash
    git clone https://github.com/shervinnd/blood-donor-predictor.git
    cd blood-donor-predictor
    ```

2.  Install dependencies:

    ``` bash
    pip install -r requirements.txt
    ```

3.  Ensure you have the dataset (`blood_donor_dataset.csv`) in the
    project directory.

## 📋 Requirements

-   pandas
-   numpy
-   scikit-learn
-   tensorflow
-   matplotlib

Install them using:

``` bash
pip install pandas numpy scikit-learn tensorflow matplotlib
```

## 🚀 Usage

1.  **Prepare the Dataset**: Place `blood_donor_dataset.csv` in the
    project root.
2.  **Run the Notebook**: Open `donor.ipynb` in Jupyter Notebook or
    Google Colab.
3.  **Train the Model**: Execute the cells to preprocess data, train the
    model, and evaluate predictions.
4.  **Visualize Results**: Check the ROC curve and prediction
    comparisons in the output.

## 📊 Model Details

-   **Input Features**:
    -   `months_since_first_donation`
    -   `number_of_donation`
    -   `pints_donated`
    -   `blood_group` (encoded)
-   **Target**: `availability` (Yes/No)
-   **Architecture**:
    -   Dense layers: 256, 128, 64, 32 units with ReLU activation
    -   Batch normalization and dropout for regularization
    -   Sigmoid output for binary classification
-   **Training**:
    -   Optimizer: Adam (learning rate = 0.001)
    -   Loss: Binary Crossentropy
    -   Epochs: 200 with early stopping and learning rate reduction

## 📈 Results

The model achieves high accuracy on the test set, with detailed
predictions compared to actual values. The ROC curve visualizes the
trade-off between true positive and false positive rates, with AUC
indicating model performance. 📉

## 🤝 Contributing

Contributions are welcome! Feel free to: - Open issues for bugs or
feature requests 🐛 - Submit pull requests with improvements 🔧 -
Suggest enhancements to the model or preprocessing pipeline 🌟

## 📜 License

This project is licensed under the MIT License. See the
[LICENSE](LICENSE) file for details.

## 📧 Contact

For questions or feedback, reach out via GitHub Issues or email at
your.email@example.com.

Happy coding and predicting! 🚀🩺

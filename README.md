# Dog_Cat_CNN
# Cat vs Dog Image Classification

This is a beginner-friendly Streamlit application that uses an existing
trained TensorFlow/Keras model to classify uploaded images as Cat or Dog.
It does not download a dataset or train a new model.

## Project structure

```
cat_dog_classifier/
├── main.py           # Main Streamlit application (UI + sections)
├── data.py           # Dataset download and preparation
├── model.py          # CNN model: build, train, save, evaluate
├── prediction.py     # Image prediction using the saved model
├── requirements.txt  # Python packages to install
└── README.md         # This file
```

## How to run the application

1. Open a terminal in the `cat_dog_classifier` folder.
2. Create a virtual environment (recommended):

   ```
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. Install the dependencies:

   ```
   pip install -r requirements.txt
   ```

4. Start the app:

   ```
   streamlit run main.py
   ```

## Model setup

Place `best_model.keras` in the same folder as `main.py`. The application
loads this file once through Streamlit caching and reuses it after refresh.

The saved model must use sigmoid binary classification:

- Probability below 0.5: Cat
- Probability 0.5 or above: Dog

Uploaded images are converted to RGB and resized to the input size required
by the saved model. The model in this project requires 128 × 128 images.

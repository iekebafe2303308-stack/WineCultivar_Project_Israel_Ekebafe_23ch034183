# Wine Cultivar Origin Prediction System

This project trains a wine cultivar classifier using the sklearn Wine dataset and provides a Flask web GUI for predictions.

## Features Used (6)
- alcohol
- malic_acid
- ash
- alcalinity_of_ash
- flavanoids
- proline

## Model Training
Open the notebook in model/model_building.ipynb and run all cells to retrain the model. The trained pipeline is saved to model/wine_cultivar_model.pkl.

## Run the Web App
1. Install dependencies from requirements.txt.
2. Start the Flask app:
   - Run app.py
3. Open the local URL shown in the terminal to use the GUI.

## Notes
- The model pipeline includes feature scaling and a Logistic Regression classifier.
- Update WineCultivar_hosted_webGUI_link.txt with real URLs when deploying.

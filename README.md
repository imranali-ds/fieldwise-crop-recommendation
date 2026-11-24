# Fieldwise · Crop recommendation

Fieldwise ranks three crop matches from seven measurements of a field. Enter soil nitrogen, phosphorus, potassium and pH, plus temperature, humidity and rainfall. The Streamlit app shows model probabilities and lets you compare each reading with the typical range for the top crop.

## Run locally

Use Python 3.12. From the repository root, run the following commands in a virtual environment. The trained model and preprocessing files are included.

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Select **Assess my field**, open **Example field**, choose a sample, then select **Recommend crop**. Select **Why this crop?** to see how the seven readings compare with the crop’s 10th–90th percentile training ranges.

## Data and evaluation

The included CSV has 2,200 rows across 22 crops, with 100 examples per crop. The notebook compares six classifiers on a reproducible stratified split of 1,760 training rows and 440 test rows. Its default experiment injects missing values and extreme readings before splitting to study preprocessing.

A tuned random forest with 100 trees correctly classified **434 of 440 held-out examples (98.64%)** in the publication run. This measures classification on the supplied dataset. Field performance, yield, and probability calibration need separate validation. The CSV does not specify nutrient units or a rainfall measurement period.

## Reproduce the project

Open [the training notebook](training_notebook.ipynb) in Jupyter with the repository root as the working directory and run all cells. It writes the trained model, scaler, label encoder, crop profiles, and training ranges to `artifacts/`, plus charts and comparison tables to `figures/` and `outputs/`.

| File | Purpose |
| --- | --- |
| [app.py](app.py) | Streamlit interface, input checks, crop ranking, and explanations |
| [Crop_recommendation.csv](Crop_recommendation.csv) | Seven numeric features and a crop label for each observation |
| [artifacts/](artifacts/) | Files loaded at startup for prediction |
| [Original Colab notebook](notebooks/original_colab.ipynb) | Earlier experiment with saved outputs and Colab-specific download cells |

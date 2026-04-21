# WHOOP Explorer
## Austin Juhl

This project analyzes personal WHOOP data to explore how physiological signals, sleep behavior, and daily habits relate to recovery. It includes a cleaning and modeling workflow based on exported WHOOP CSV files, plus a Streamlit app for interactive exploration.

## Project Goals

- clean and align raw WHOOP exports across cycles, sleep, journal entries, and workouts
- explore trends, distributions, correlations, and habit effects for recovery-related variables
- model next-day change in recovery score using physiological and lagged features
- present the analysis in an interactive dashboard

## Files

- `initial_analysis.ipynb`: original notebook analysis
- `whoop_analysis.py`: reusable data cleaning, diagnostics, and modeling functions
- `streamlit_app.py`: Streamlit interface with home, data analysis, and model diagnostics pages
- `physiological_cycles.csv`, `journal_entries.csv`, `workouts.csv`: exported WHOOP data
- `slides.pptx`: slides describing the project with some extra data and model interpretation information

## Live Demo
Video Demo: https://youtu.be/m8Qw4tkQv4Q
  
## Data Pipeline

The workflow converts cycle timestamps to UTC, trims all tables to the earliest date with valid blood oxygen and skin temperature values, removes large tracking-gap artifacts, merges journal and workout signals into cycle-level records, and fills remaining gaps through interpolation plus forward/backward filling. For modeling, the cleaned timeline is then segmented into continuous stretches of data so lagged features and next-day targets are only created within uninterrupted tracking windows. This avoids treating observations on opposite sides of a long gap as if they were consecutive days, which would otherwise leak unrealistic temporal relationships into the model.

## Modeling

The predictive task is next-day change in recovery score. Features include current-day physiological measures and several lagged variables. To preserve valid time ordering, lag features and targets are generated separately inside each continuous data segment rather than across breaks in tracking. The app compares:

- Random Forest
- Linear Regression on normalized inputs
- a naive baseline that predicts zero change

Model diagnostics include actual vs predicted recovery, residual behavior over time, prediction scatterplots, residual distributions, and feature importance or coefficient plots.

## Streamlit App

The app provides:

- a home page summarizing the project and cleaned dataset
- a data analysis page for interactive variable diagnostics across selectable time windows
- a model diagnostics page for comparing model behavior on train and test splits

## Running Locally

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Launch the app:

```bash
streamlit run streamlit_app.py
```

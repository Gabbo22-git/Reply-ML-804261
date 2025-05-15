# Forecasting the unexpected

* Group Name: Napoletani a Roccaraso
* Group Member: Granata Annalaura, Lupico Daniele, Rizzo Gabriele

---

## Table of Contents
- [Forecasting the unexpected](#forecasting-the-unexpected)
  - [Table of Contents](#table-of-contents)
  - [\[Section 1\] Introduction](#section-1-introduction)
  - [\[Section 2\] Methods](#section-2-methods)
    - [Project Overview](#project-overview)
    - [Dataset Description](#dataset-description)
      - [Initial Dataset (`FullDayWithAlarms.csv`)](#initial-dataset-fulldaywithalarmscsv)
      - [Data Augmentation (`week_dataset.csv`)](#data-augmentation-week_datasetcsv)
    - [Data Preprocessing](#data-preprocessing)
    - [Feature Engineering and Selection](#feature-engineering-and-selection)
    - [Algorithms](#algorithms)
      - [Time Series Forecasting Models (LSTM \& TCN)](#time-series-forecasting-models-lstm--tcn)
      - [Anomaly Detection Model (Autoencoder)](#anomaly-detection-model-autoencoder)
    - [Training Overview](#training-overview)
      - [Data Splitting](#data-splitting)
      - [Forecasting Model Training (LSTM \& TCN)](#forecasting-model-training-lstm--tcn)
      - [Autoencoder Model Training](#autoencoder-model-training)
      - [Threshold Selection for Autoencoder](#threshold-selection-for-autoencoder)
    - [Development Environment and Reproducibility](#development-environment-and-reproducibility)
    - [Step-by-Step Usage Guide](#step-by-step-usage-guide)
  - [\[Section 3\] Experimental Design](#section-3-experimental-design)
    - [Experiment 1: Forecasting Model Performance Comparison](#experiment-1-forecasting-model-performance-comparison)
    - [Experiment 2: Autoencoder-based Anomaly Detection Evaluation](#experiment-2-autoencoder-based-anomaly-detection-evaluation)
    - [Experiment 3: Autoencoder Threshold Optimization](#experiment-3-autoencoder-threshold-optimization)
  - [\[Section 4\] Results](#section-4-results)
    - [Main Findings](#main-findings)
    - [Detailed Interpretation of Results](#detailed-interpretation-of-results)
      - [Data Augmentation Insights](#data-augmentation-insights)
      - [Forecasting Models (LSTM \& TCN)](#forecasting-models-lstm--tcn)
      - [Autoencoder Anomaly Detection](#autoencoder-anomaly-detection)
    - [Code Insights](#code-insights)
    - [Figures and Tables of Results](#figures-and-tables-of-results)
  - [\[Section 5\] Conclusions](#section-5-conclusions)
    - [Take-away Points](#take-away-points)
    - [Limitations and Problems Encountered](#limitations-and-problems-encountered)
    - [Future Work](#future-work)

---

## [Section 1] Introduction

This project focuses on developing and evaluating an anomaly detection system for time-series data representing transactional metrics of a service or application. The core objective is to identify unusual patterns or deviations from normal operational behavior, which could signify performance issues, errors, or other critical incidents. The input data, initially a single day's worth of minute-by-minute aggregated transactions, is augmented to create a more robust dataset for model training and evaluation.

We explore a two-pronged approach:
1.  **Time Series Forecasting:** Using Long Short-Term Memory (LSTM) networks and Temporal Convolutional Networks (TCN) to predict key transactional metrics. While primarily for forecasting, the reconstruction/prediction errors from such models can also be indicative of anomalies.
2.  **Dedicated Anomaly Detection:** Employing an LSTM-based Autoencoder, specifically trained on normal operational data, to identify anomalies based on reconstruction errors.

This report details the entire workflow, from initial data exploration and augmentation, preprocessing, model design and training, to a thorough experimental evaluation and interpretation of the results. The goal is to build an effective anomaly detection system and understand the efficacy of different modeling choices for this specific type of transactional data.

---

## [Section 2] Methods

This section describes the proposed ideas, design choices, algorithms, training overview, and the development environment, enabling a reader to understand the design decisions and reproduce the project.

### Project Overview

The anomaly detection system follows these general steps:

The project workflow can be summarized as follows:

1.  **Raw Single-Day Data (.csv)**
    *   Input: `FullDayWithAlarms.csv`
    *   Description: Transactional data aggregated per minute for a single day.
    *   ⬇️
2.  **Initial EDA & Preprocessing**
    *   Tasks: Data cleaning, type conversion (e.g., `data_ora` to datetime), understanding initial feature distributions, dropping constant columns.
    *   Output: Cleaned single-day DataFrame.
    *   ⬇️
3.  **Data Augmentation**
    *   Method: Generate 6 synthetic days based on minute-by-minute statistics from the original day. Inject controlled anomalies (spikes/drops in `numero_transazioni`).
    *   Output: `week_dataset.csv` (7 days of data, including original day + 6 synthetic days with anomalies).
    *   ⬇️
4.  **Feature Engineering**
    *   Tasks: Extract time-based features (e.g., `hour`, `day_of_week`) from the datetime index.
    *   Output: Enriched DataFrame.
    *   ⬇️
5.  **Data Splitting (Chronological)**
    *   Method: Split the 7-day dataset into Training (5 days), Validation (1 day), and Test (1 day) sets.
    *   Output: `train_data`, `val_data`, `test_data` DataFrames.
    *   ⬇️
    *   **Path A: Forecasting**
        1.  **Forecasting Models Training (LSTM, TCN)**
            *   Input: Scaled sequences from *normal* training data.
            *   Objective: Predict `numero_transazioni` and `numero_transazioni_errate`.
            *   ⬇️
        2.  **Forecasting Evaluation**
            *   Metrics: MSE, MAE, RMSE on the test set.
            *   ⬇️
    *   **Path B: Anomaly Detection**
        1.  **Anomaly Detection Model Training (Autoencoder)**
            *   Input: Scaled sequences from *normal* training data.
            *   Objective: Learn to reconstruct normal patterns.
            *   ⬇️
        2.  **Anomaly Detection Evaluation**
            *   Method: Calculate reconstruction error on test set; use an optimized threshold (from validation set) to classify anomalies.
            *   Metrics: Precision, Recall, F1-Score, AUC-ROC, AUC-PR.
            *   ⬇️
6.  **Insights & Reporting**
    *   Tasks: Consolidate results, compare model performances, draw conclusions, identify limitations, and suggest future work.

### Dataset Description

#### Initial Dataset (`FullDayWithAlarms.csv`)
The project started with a dataset named `FullDayWithAlarms.csv`, containing transactional data aggregated over 60-second intervals for one full day (2024-05-31, from 07:01:11 to 23:59:23).

**Original Features:**
*   `COD GIOCO`: Identifier for the system/machine (constant value 8).
*   `DATA ORA`: Timestamp of the data record.
*   `INTERVALLO ACQUISIZIONE`: Acquisition interval in seconds (constant value 60).
*   `NUMERO TRANSAZIONI`: Number of transactions in the interval.
*   `TEMPO MIN`: Minimum transaction time in the interval.
*   `TEMPO MAX`: Maximum transaction time in the interval.
*   `TEMPO MEDIO`: Average transaction time in the interval.
*   `NUMERO RETRY`: Number of retries in the interval.
*   `NUMERO TRANSAZIONI ERRATE`: Number of errored transactions in the interval.

**Initial EDA Findings on Original Day:**
*   `COD GIOCO` and `INTERVALLO ACQUISIZIONE` were constant and thus dropped.
*   `NUMERO TRANSAZIONI` showed typical daily patterns with peaks and troughs.
*   `TEMPO MEDIO` generally followed transaction volume but had some spikes.
*   `NUMERO RETRY` was mostly zero but showed significant spikes, indicating potential issues. (See `initial_timeseries_overview.png`).
*   `TEMPO MAX` had an extreme outlier (approx 25 hours), suggesting data quality issues or a very specific event not representative of typical operations. `TEMPO MIN` was consistently low. For these reasons, and to simplify the feature set for initial modeling, `TEMPO MIN` and `TEMPO MAX` were used during the data augmentation phase primarily for clipping `TEMPO MEDIO` to realistic bounds, and then dropped.

`![Initial Time-Series Overview](images/initial_timeseries_overview.png)`
*Figure 2: Overview of key metrics from the original single-day dataset.*

#### Data Augmentation (`week_dataset.csv`)
To create a more substantial and diverse dataset for robust model training, especially for learning normal patterns and injecting anomalies, the single day of data was augmented to simulate a full week (7 days).
The process involved:
1.  **`build_minute_stats`:** Calculating minute-by-minute statistics (min, max, mean, std) for each relevant feature from the original day. This captures the typical behavior for each minute of a "standard" day. The period from 23:00-23:59 was used to extrapolate data for 00:00-02:59 to cover the early morning hours.
2.  **`synth_day`:** Generating 6 synthetic days. For each minute of a synthetic day:
    *   Values for features were sampled from a normal distribution parameterized by the mean and standard deviation derived in `build_minute_stats` for that specific minute of the day.
    *   Scaling factors (randomly chosen within a range, e.g., 0.90-1.05) and noise were added to introduce variability. One "slow" day (day 3) used a slightly different scaling range (0.70-0.85) to simulate lower activity.
    *   Timestamps were adjusted for each new day.
    *   **Anomaly Injection:** Anomalies were synthetically injected into the `numero_transazioni` feature.
        *   The probability of an anomaly occurring was scaled based on the time of day (higher during busy hours 11:00-14:00 and 17:00-21:00, lower during early morning 00:00-02:00).
        *   Anomalies could be either spikes (increasing transaction count by 10-25%) or drops (decreasing transaction count by 5-15%), with spikes being more common (90% chance).
        *   The `is_anomaly` flag was set to 1 for these injected points.
3.  **`make_week`:** Concatenating the original day (with `is_anomaly=0`) and the 6 synthetic days.

This augmentation process resulted in `week_dataset.csv`, containing 8206 data points spanning from 2024-05-31 07:01:11 to 2024-06-07 02:59:00. The primary goal was to have sufficient data to train models to recognize "normal" patterns and to have labeled anomalies for supervised evaluation of the anomaly detection phase.

`![Augmented Transactions with Anomalies](images/augmented_transactions_with_anomalies.png)`
*Figure 3: `numero_transazioni` showing the original day (orange) followed by augmented days (blue) with injected anomalies (red dots).*

The distributions of the augmented data were compared to the original day's data to ensure the augmentation process maintained reasonable characteristics while introducing necessary variance.

`![Distribution of numero_transazioni: Original vs. Augmented](images/dist_numero_transazioni_orig_vs_aug.png)`
*Figure 4: Comparison of `numero_transazioni` distribution.*

`![Distribution of tempo_medio: Original vs. Augmented](images/dist_tempo_medio_orig_vs_aug.png)`
*Figure 5: Comparison of `tempo_medio` distribution.*

`![Distribution of numero_retry: Original vs. Augmented](images/dist_numero_retry_orig_vs_aug.png)`
*Figure 6: Comparison of `numero_retry` distribution.*

`![Distribution of numero_transazioni_errate: Original vs. Augmented](images/dist_numero_transazioni_errate_orig_vs_aug.png)`
*Figure 7: Comparison of `numero_transazioni_errate` distribution.*

These plots confirmed that the augmentation process, while adding more data points and variability (especially for `numero_retry` and `tempo_medio` which have more sparse extreme values in the original day), generally preserved the shape and scale of the original distributions.

### Data Preprocessing
The following preprocessing steps were applied to the `week_dataset.csv` within the `load_and_preprocess_data` function (from `utils.py`, called in `prepare_data.py`):
1.  **Datetime Conversion:** The `data_ora` column was converted to `datetime` objects and set as the DataFrame index. This is crucial for time-series analysis.
2.  **Feature Selection (Initial):**
    *   The `tempo_min` and `tempo_max` columns were dropped after the augmentation process, as `tempo_medio` was deemed more representative for per-minute analysis, and the min/max values were primarily used to ensure `tempo_medio` stayed within realistic bounds during synthetic data generation.
    *   The final features used for modeling are: `numero_transazioni`, `tempo_medio`, `numero_retry`, and `numero_transazioni_errate`.

### Feature Engineering and Selection
1.  **Time-based Features:** In `prepare_data.py`, the following time-based features were engineered from the `data_ora` index:
    *   `hour`: The hour of the day.
    *   `day_of_week`: The day of the week (0=Monday, 6=Sunday).
    These features were added to potentially help models capture daily and weekly seasonality, although the primary models (LSTM, TCN, Autoencoder) are designed to learn temporal patterns directly from sequences of the core transactional features.
2.  **Feature Scaling:**
    *   **`StandardScaler`** from `sklearn.preprocessing` was used. This scaler standardizes features by removing the mean and scaling to unit variance.
    *   **Why `StandardScaler`?** Many machine learning algorithms, especially neural networks and distance-based methods, perform better when input numerical attributes have a similar scale. Standardization helps prevent features with larger values from dominating the learning process.
    *   Separate scalers were fit:
        *   `scaler_forecast.joblib`: Fit *only* on the **normal** data from the **training set** for forecasting tasks. This ensures the model learns the characteristics of normal operational data.
        *   `scaler_ad.joblib`: Fit *only* on the **normal** data from the **training set** for the Autoencoder anomaly detection task. This is critical because the autoencoder must learn to reconstruct "normal" patterns accurately.
    *   The validation and test sets were transformed using the scaler fitted on the training data to prevent data leakage.

3.  **Feature Selection (Modeling):**
    *   The core transactional features selected for modeling were: `numero_transazioni`, `tempo_medio`, `numero_retry`, `numero_transazioni_errate`.
    *   The time-based features (`hour`, `day_of_week`) were included in the dataframe passed to the scaling and sequencing steps, making them available to the models if their architecture was designed to use them explicitly (though the sequence-based models primarily focus on the patterns within the sequences of the core features).
    *   A correlation matrix was plotted to understand linear relationships between these features.

    `![Feature Correlation Matrix](images/feature_correlation_matrix.png)`
    *Figure 8: Correlation matrix of the selected features. `tempo_medio` and `numero_retry` show a notable positive correlation (0.86), suggesting that higher average times might coincide with more retries, which is intuitively plausible.*

### Algorithms

#### Time Series Forecasting Models (LSTM & TCN)
Two types of models were developed for forecasting `numero_transazioni` and `numero_transazioni_errate`. These models were trained only on data points labeled as normal to learn the baseline behavior.
The motivation for forecasting was twofold: first, as a standard time-series task, and second, because significant deviations between forecasted and actual values can themselves be an indicator of anomalies.

1.  **LSTM (Long Short-Term Memory Network):**
    *   **Description:** LSTMs are a type of Recurrent Neural Network (RNN) well-suited for learning long-range dependencies in sequential data. They use gating mechanisms (input, forget, output gates) to control the flow of information, mitigating the vanishing gradient problem common in simple RNNs.
    *   **Architecture (from `train_forecasting.py`):**
        *   Input Layer: Sequences of shape (`SEQUENCE_LENGTH`, `num_features`). `SEQUENCE_LENGTH` is 60 (minutes).
        *   Two Bidirectional LSTM layers:
            *   1st BiLSTM: 64 units, `return_sequences=True`, dropout (0.3).
            *   Batch Normalization.
            *   2nd BiLSTM: 32 units (64/2), `return_sequences=False`, dropout (0.3).
            *   Batch Normalization.
        *   Dense Layers:
            *   Dense: 32 units, ReLU activation.
            *   Dropout (0.3).
            *   Dense: 16 units (64/4), ReLU activation.
            *   Dropout (0.3).
        *   Output Layer: Dense layer with `len(PREDICTION_FEATURES)` units (2 units for `numero_transazioni` and `numero_transazioni_errate`), linear activation.
    *   **Why this design?**
        *   Bidirectional LSTMs process sequences in both forward and backward directions, allowing the model to capture context from past and future (within the input sequence).
        *   Stacking LSTM layers allows for learning hierarchical representations of temporal features.
        *   Batch Normalization helps stabilize training and can speed up convergence.
        *   Dropout is used for regularization to prevent overfitting.
        *   Dense layers after LSTMs act as a final regressor.
    *   **Optimizer:** Adam (`learning_rate=0.001`).
    *   **Loss Function:** Huber loss, which is less sensitive to outliers than Mean Squared Error, making it more robust for time series data that might contain occasional spikes.

2.  **TCN (Temporal Convolutional Network):**
    *   **Description:** TCNs use causal convolutions (output at time `t` depends only on inputs at time `t` and earlier) and dilated convolutions to achieve large receptive fields with fewer layers compared to standard CNNs. They often train faster and perform competitively with RNNs on sequence modeling tasks.
    *   **Architecture (from `train_forecasting.py`):**
        *   Input Layer: Sequences of shape (`SEQUENCE_LENGTH`, `num_features`).
        *   Initial Conv1D layer (64 filters, kernel size 4, causal padding).
        *   Batch Normalization.
        *   Stack of Residual Blocks: Each block consists of two Conv1D layers with ReLU activation, Batch Normalization, and Dropout (0.1). Dilated convolutions are used with rates [1, 2, 4, 8]. Skip connections are used within each residual block.
        *   Global Average Pooling 1D layer.
        *   Dense Layers:
            *   Dense: 32 units (64/2), ReLU activation.
            *   Batch Normalization, Dropout (0.2).
            *   Dense: 16 units (64/4), ReLU activation.
            *   Batch Normalization, Dropout (0.2).
        *   Output Layer: Dense layer with `len(PREDICTION_FEATURES)` units, linear activation.
    *   **Why this design?**
        *   Causal convolutions ensure no leakage from future to past.
        *   Dilated convolutions allow the network to have a receptive field that grows exponentially with depth, capturing long-range dependencies efficiently.
        *   Residual connections help in training deeper networks by mitigating vanishing gradients.
    *   **Optimizer:** Adam (`learning_rate=0.001`).
    *   **Loss Function:** Huber loss.

#### Anomaly Detection Model (Autoencoder)
An LSTM-based Autoencoder was chosen as the primary model for anomaly detection.
*   **Description:** Autoencoders are neural networks trained to reconstruct their input. For anomaly detection, they are trained exclusively on "normal" data. The idea is that the model will learn to reconstruct normal patterns well (low reconstruction error), but will struggle to reconstruct anomalous patterns (high reconstruction error).
*   **Architecture (from `train_anomaly_detection.py`):**
    *   **Encoder:**
        *   Input Layer: Sequences of shape (`SEQUENCE_LENGTH`, `num_features`).
        *   Bidirectional LSTM (128 units, `return_sequences=True`, dropout 0.2).
        *   Batch Normalization.
        *   Bidirectional LSTM (64 units, `return_sequences=False`, dropout 0.2).
        *   Batch Normalization.
        *   Dense Bottleneck (32 units, ReLU activation).
        *   Batch Normalization.
    *   **Decoder:**
        *   Dense (64 units, ReLU activation).
        *   Batch Normalization.
        *   `RepeatVector` to repeat the bottleneck representation `SEQUENCE_LENGTH` times.
        *   Bidirectional LSTM (64 units, `return_sequences=True`, dropout 0.2).
        *   Batch Normalization.
        *   Bidirectional LSTM (128 units, `return_sequences=True`, dropout 0.2).
        *   Batch Normalization.
        *   `TimeDistributed(Dense(num_features, activation='linear'))` to reconstruct the input sequence.
*   **Why this design?**
    *   The LSTM layers are capable of capturing temporal dependencies within the input sequences. Bidirectional LSTMs enhance this by considering context from both past and future elements within the sequence.
    *   The encoder compresses the input sequence into a lower-dimensional representation (bottleneck), forcing it to learn the most salient features of normal data.
    *   The decoder attempts to reconstruct the original sequence from this compressed representation.
    *   Batch Normalization and Dropout are used for regularization and training stability.
*   **Optimizer:** Adam (`learning_rate=0.001`).
*   **Loss Function:** Mean Squared Error (MSE), as the goal is to minimize the difference between the input and its reconstruction. `mae` (Mean Absolute Error) was also monitored.

### Training Overview

#### Data Splitting
The augmented `week_dataset.csv` (7 days of data) was split chronologically:
*   **Training Set:** First 5 days (from `TRAIN_SIZE` in `config.py`).
*   **Validation Set:** Next 1 day (from `VAL_SIZE` in `config.py`).
*   **Test Set:** Last 1 day (from `TEST_SIZE` in `config.py`).

**Why chronological split?** For time-series data, it's crucial to validate and test on data that occurs after the training data to simulate a real-world scenario where the model predicts future, unseen data. Random splitting would lead to data leakage and overly optimistic performance estimates.

The distribution of features and anomalies across these splits was visualized:
`![Feature Distributions Across Splits](images/split_feature_distributions.png)`
*Figure 9: Distribution of key features across Train, Validation, and Test sets.*

`![Feature Time Series Across Splits](images/split_feature_timeseries.png)`
*Figure 10: Time series of key features showing the Train, Validation, and Test splits.*

`![Anomaly Distribution Across Splits](images/anomaly_distribution_splits.png)`
*Figure 11: Distribution of normal (0) vs. anomalous (1) samples in Train, Validation, and Test sets. Note that anomalies are primarily in Val/Test for evaluation purposes, as forecasting and autoencoder models are trained on normal data.*

#### Forecasting Model Training (LSTM & TCN)
*   **Input Data:** Sequences of scaled features (`X_train_forecast`, `X_val_forecast`) of length `SEQUENCE_LENGTH` (60).
*   **Target Data:** The next time step's scaled values for `numero_transazioni` and `numero_transazioni_errate` (`y_train_forecast`, `y_val_forecast`).
*   **Training Data:** Only sequences derived from *normal* data (where `is_anomaly == 0`) in the training set were used. This is to ensure the forecasting models learn the baseline behavior of the system under normal conditions.
*   **Epochs:** 50 (as per `LSTM_PARAMS['epochs']` / `TCN_PARAMS['epochs']`).
*   **Batch Size:** 64.
*   **Callbacks:**
    *   `EarlyStopping`: Monitored `val_loss` with patience of 15 (LSTM) / 10 (TCN) to prevent overfitting and restore best weights.
    *   `ModelCheckpoint`: Saved the best model based on `val_loss`.
    *   `ReduceLROnPlateau`: Reduced learning rate if `val_loss` plateaued.
    *   `LearningRateScheduler`: Custom learning rate schedule.
    *   `TensorBoard`: For monitoring training progress.

#### Autoencoder Model Training
*   **Input Data:** Sequences of scaled features (`X_train_ad`, `X_val_ad`) of length `SEQUENCE_LENGTH` (60). The target for the autoencoder is to reconstruct its own input.
*   **Training Data:** Only sequences derived from *normal* data (where `is_anomaly == 0`) in the training set were used. This is fundamental for autoencoder-based anomaly detection, as the model must learn to reconstruct "normal" well.
*   **Epochs:** 50 (as per `AUTOENCODER_PARAMS['epochs']`).
*   **Batch Size:** 64 (or smaller if training data is limited, dynamically set to `min(128, len(X_train))`).
*   **Callbacks:** Similar to forecasting models, including `EarlyStopping` (patience 10), `ModelCheckpoint`, and `ReduceLROnPlateau`.

#### Threshold Selection for Autoencoder
The autoencoder outputs a reconstruction of its input. Anomalies are identified when the reconstruction error (e.g., Mean Squared Error - MSE) between the input and its reconstruction is high.
1.  **Initial Threshold Estimation:** The model was applied to the *validation set* (which contains both normal and synthetically injected anomalous data). Reconstruction errors (MSE) were calculated for each validation sequence.
2.  **Percentile-based Threshold:** An initial threshold was determined as a high percentile (e.g., 95th percentile as per `AUTOENCODER_PARAMS['reconstruction_threshold_percentile']`) of the reconstruction errors on the *normal* validation data (or all validation data if only a single threshold is to be derived this way).
3.  **F1-Score Optimization:** To refine the threshold, different percentile values of the validation errors were tested ([75, 80, 85, 90, 92, 95, 97, 98, 99, 99.5]). For each resulting threshold, Precision, Recall, and F1-score were calculated on the validation set (using the known `is_anomaly` labels). The threshold that maximized the F1-score was chosen as the final operational threshold. This aims to find a good balance between correctly identifying anomalies (Recall) and minimizing false alarms (Precision). The analysis was saved in `autoencoder_threshold_analysis.csv`.

### Development Environment and Reproducibility
The project was developed using Python 3.11.11 with a conda environment (the only way to run tensorflow on a MacBook Air M1 2020). Key libraries include:
*   `pandas`: For data manipulation and analysis.
*   `numpy`: For numerical operations.
*   `scikit-learn`: For `StandardScaler` and evaluation metrics.
*   `tensorflow` (with Keras API): For building and training LSTM, TCN, and Autoencoder models.
*   `matplotlib` & `seaborn`: For data visualization.
*   `joblib`: For saving/loading scalers.

To ensure reproducibility, the environment can be recreated using `conda` or `pip`.

**Using Conda:**
1.  Ensure Anaconda or Miniconda is installed.
2.  Create an environment from `environment.yml`:
    ```bash
    conda env create -f environment.yml
    conda activate your_env_name
    ```

**Using Pip:**
1.  Ensure Python and pip are installed.
2.  Install dependencies from `requirements.txt` (generate with `pip freeze > requirements.txt`):
    ```bash
    conda activate env_name
    pip install -r requirements.txt
    ```

### Step-by-Step Usage Guide
1.  **Clone the Repository:**
    ```bash
    git clone [URL_OF_YOUR_REPOSITORY]
    cd [REPOSITORY_NAME]
    ```
2.  **Set up Environment:** Follow the instructions in the [Development Environment and Reproducibility](#development-environment-and-reproducibility) section.
3.  **Dataset:**
    *   The initial raw dataset (`FullDayWithAlarms.csv`) should be placed in a relevant location if not included directly (e.g., a `raw_data` folder). The script assumes `week_dataset.csv` will be generated or is available in the `data/` directory (as per `DATA_FILE` in `config` cell).
    *   The `Phase0 data preparation` script will process `week_dataset.csv` (or generate it if the augmentation part is integrated there) and save processed files into the `data/` directory.
4.  **Run the Main Notebook (`main.ipynb`):**
    *   Open `main.ipynb` using Jupyter Notebook, JupyterLab, or VS Code.
    *   The notebook should ideally guide through:
        *   Data loading and initial EDA (as shown in your provided code).
        *   Data augmentation steps (or loading the augmented `week_dataset.csv`).
        *   Run the cells `config` and `utils`.
        *   Running `Phase0 data preparation` logic to split, scale, and sequence data.
        *   Running `Phase1 forecasting` logic to train and evaluate LSTM and TCN.
        *   Running `Phase2 Anomaly Detection` logic to train and evaluate the Autoencoder.
        *   Running `Phase3 Evaluation` logic to generate final comparison plots and summaries.
    *   Execute cells sequentially. Text cells will describe the code and its expected output.
5.  **Outputs:**
    *   Metrics, plots, and model files will be saved in the `results/` and `models/` directories as configured in `config` cell.
    *   Key visualizations for this README are expected to be in the `images/` folder.

---

## [Section 3] Experimental Design

To validate the contributions of this project, several experiments were conducted:

### Experiment 1: Forecasting Model Performance Comparison
*   **Main Purpose:** To evaluate and compare the performance of LSTM and TCN models in forecasting `numero_transazioni` and `numero_transazioni_errate` on unseen test data. This helps establish a baseline understanding of how well normal system behavior can be predicted.
*   **Baseline(s):** The LSTM and TCN models are compared against each other.
*   **Evaluation Metrics(s):**
    *   **Mean Squared Error (MSE):** Measures the average squared difference between actual and predicted values. Sensitive to large errors.
        *   *Why:* Standard regression metric, penalizes larger errors more heavily.
    *   **Mean Absolute Error (MAE):** Measures the average absolute difference. Less sensitive to outliers than MSE.
        *   *Why:* Provides a more direct interpretation of average error magnitude.
    *   **Root Mean Squared Error (RMSE):** Square root of MSE, bringing the metric back to the original unit of the target variable.
        *   *Why:* Commonly used, interpretable in the same units as the target.
    *   *These metrics were calculated on the scaled data and potentially on denormalized data for better interpretability.*

### Experiment 2: Autoencoder-based Anomaly Detection Evaluation
*   **Main Purpose:** To assess the effectiveness of the LSTM-based Autoencoder in distinguishing between normal and synthetically injected anomalous data points in the test set.
*   **Baseline(s):** While no other explicit anomaly detection algorithm was implemented for direct comparison in this phase, the performance is judged on its ability to correctly classify the labeled anomalies provided through data augmentation.
*   **Evaluation Metrics(s):** (Calculated on the test set, using the optimal threshold from Experiment 3)
    *   **Precision:** Of all instances flagged as anomalies, what fraction were actual anomalies? (TP / (TP + FP))
        *   *Why:* Important to minimize false alarms, which can lead to alert fatigue.
    *   **Recall (Sensitivity):** Of all actual anomalies, what fraction were correctly identified? (TP / (TP + FN))
        *   *Why:* Crucial for ensuring that most true anomalies are detected.
    *   **F1-Score:** The harmonic mean of Precision and Recall (2 * (Precision * Recall) / (Precision + Recall)).
        *   *Why:* Provides a single score that balances Precision and Recall, especially useful for imbalanced datasets like anomaly detection.
    *   **ROC AUC (Area Under the Receiver Operating Characteristic Curve):** Measures the model's ability to distinguish between classes across all possible thresholds.
        *   *Why:* Threshold-independent measure of separability. An AUC of 0.5 is random, 1.0 is perfect.
    *   **Precision-Recall AUC (Average Precision):** Area under the Precision-Recall curve.
        *   *Why:* More informative than ROC AUC for highly imbalanced datasets, as it focuses on the performance on the minority (anomalous) class.
    *   **Confusion Matrix:** A table showing True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN).
        *   *Why:* Provides a detailed breakdown of classification performance and types of errors.

### Experiment 3: Autoencoder Threshold Optimization
*   **Main Purpose:** To determine the optimal reconstruction error threshold for the Autoencoder that best separates normal from anomalous data on the validation set, balancing precision and recall.
*   **Baseline(s):** N/A; this is an internal optimization step for the Autoencoder.
*   **Evaluation Metrics(s):**
    *   Precision, Recall, and F1-Score were calculated for various threshold percentiles (derived from validation set reconstruction errors) when applied to the validation set itself (which contains known normal and anomalous samples).
    *   The threshold maximizing the F1-score (or another chosen balance like prioritizing recall) was selected for final evaluation on the test set.
        *   *Why:* Threshold selection is critical for binary classification based on anomaly scores. Optimizing on a validation set helps find a data-driven threshold.

---

## [Section 4] Results

This section describes the main findings from the experiments, including visual and tabular results. All figures presented here were generated from the code in `main.ipynb`.

### Main Findings

**Data Augmentation & Preparation:**
*   The data augmentation process successfully expanded the single-day dataset into a week-long dataset, allowing for more robust model training and evaluation of anomaly detection capabilities.
*   The distribution plots (Figures 4-7) show that the augmented data, while introducing more variance and anomalies, generally maintained the characteristic patterns of the original data for key features.
*   The chronological split ensured a realistic evaluation scenario, with distinct distributions for train, validation, and test sets as seen in Figures 9 and 11.

**Forecasting Models (LSTM & TCN):**
*   Both LSTM and TCN models were trained to forecast `numero_transazioni` and `numero_transazioni_errate`.
*   Qualitatively, both models learned the general daily patterns of `numero_transazioni` but struggled more with the sparser and more volatile `numero_transazioni_errate` (See Figures `lstm_forecast_vs_actual.png` and `tcn_forecast_vs_actual.png`).
*   The error distributions (Figures `lstm_error_distribution.png` and `tcn_error_distribution.png`) were generally centered around zero but showed some skewness, indicating areas where predictions were consistently over or under.
*   Quantitatively, the TCN model slightly outperformed the LSTM model on the test set across MSE, MAE, and RMSE metrics for predicting the selected features. *(Refer to Table 1 and Figure `forecasting_model_comparison.png`)*.

**Autoencoder Anomaly Detection:**
*   The LSTM-based Autoencoder was trained on normal data and then evaluated on its ability to detect injected anomalies in the test set using a threshold optimized on the validation set.
*   The optimal threshold (maximizing F1-score on validation) was found to be [Value from your `autoencoder_threshold_analysis.csv`, e.g., ~0.3 based on a quick look at the provided CSV].
*   On the test set, the Autoencoder achieved:
    *   Precision: 0.984
    *   Recall: 0.294
    *   F1-Score: 0.453
    *   AUC-ROC: 0.830
    *   Average Precision (AUC-PR): 0.937
    *(These values are taken directly from your `anomaly_metrics.csv`)*
*   The anomaly scores distribution (Figure `autoencoder_anomaly_scores_distribution.png`) shows a noticeable difference in distributions between normal and anomalous samples, though with some overlap.
*   The Confusion Matrix (Figure `autoencoder_confusion_matrix.png`) indicates a high number of true negatives and a good number of true positives, but also a significant number of false negatives (anomalies missed). False positives were very low.

### Detailed Interpretation of Results

#### Data Augmentation Insights
The augmentation was crucial. Without it, training robust models, especially an autoencoder on "normal" behavior and evaluating anomaly detection on unseen anomalies, would be very difficult with just a single day of data.
*   **What was done:** A single day of transactional data was used to generate statistical profiles for each minute. Six additional synthetic days were created by sampling from these profiles, adding noise, scaling, and injecting anomalies (spikes/drops in transaction counts).
*   **Why it was done:** To create a larger dataset for training, introduce variability, and provide labeled anomalies for evaluating the detection performance.
*   **What it helped discover/served:** It enabled the training of more complex models like LSTMs and Autoencoders and provided a basis for quantitative evaluation of anomaly detection. The plots comparing original and augmented distributions (Figures 4-7) confirm that the augmentation, while adding diversity, did not drastically alter the fundamental data characteristics. Figure 3 clearly shows the injected anomalies, which serve as ground truth for the Autoencoder evaluation.

#### Forecasting Models (LSTM & TCN)
*   **What was done:** LSTM and TCN models were trained to predict `numero_transazioni` and `numero_transazioni_errate` based on the preceding 60 minutes of data. They were trained only on normal data.
*   **Why it was done:** To establish if these complex time-series models could accurately capture the normal operational patterns. While not the primary anomaly detection method here, large forecasting errors can signal anomalies.
*   **What it helped discover/served:**
    *   Both models learned the cyclical patterns in `numero_transazioni` reasonably well (Figures `lstm_forecast_vs_actual.png`, `tcn_forecast_vs_actual.png`). The predictions follow the general trend of actual values.
    *   Predicting `numero_transazioni_errate` was more challenging due to its more erratic nature and lower values; the models tended to predict a smoothed-out version.
    *   The TCN model showed slightly better performance in terms of MSE, MAE, and RMSE (Table 1, Figure `forecasting_model_comparison.png`), suggesting its architecture might be slightly more adept at capturing the specific temporal dependencies in this dataset for forecasting.
    *   The training history (Figure `forecasting_training_history.png`) shows both models converging, with validation loss generally tracking training loss, indicating no severe overfitting.
    *   Future predictions (Figures `lstm_future_forecast.png`, `tcn_future_forecast.png`) show the models attempting to extrapolate patterns, which is inherently difficult for longer horizons.

#### Autoencoder Anomaly Detection
*   **What was done:** An LSTM-based Autoencoder was trained on sequences of normal data. Anomalies were then identified in the test set by flagging data points where the reconstruction MSE exceeded a dynamically determined threshold.
*   **Why it was done:** Autoencoders are a common unsupervised/semi-supervised approach for anomaly detection. By learning to reconstruct only normal data well, they should produce higher errors for abnormal data.
*   **What it helped discover/served:**
    *   **Thresholding is Key:** The threshold analysis (Table 3, based on `autoencoder_threshold_analysis.csv`) clearly demonstrates the trade-off between precision and recall. A lower threshold increases recall (catches more anomalies) but decreases precision (more false alarms). The F1-score was maximized at a threshold of [Value from your CSV, e.g., ~0.3], which was used for the final evaluation.
    *   **Performance:**
        *   The Confusion Matrix (Figure `autoencoder_confusion_matrix.png`) shows that the model correctly identified 248 anomalies (TP) but missed 596 (FN). It had very few false positives (4 FP).
        *   This leads to a high Precision (0.984) because when it flags an anomaly, it's almost always correct. However, the Recall (0.294) is low, meaning it misses a significant portion of actual anomalies.
        *   The F1-Score (0.453) reflects this imbalance.
        *   The AUC-ROC of 0.830 (Figure `autoencoder_roc_curve.png`) indicates a good level of separability between normal and anomalous samples across various thresholds, better than random chance.
        *   The AUC-PR of 0.937 (Figure `autoencoder_pr_curve.png`) is high, which is encouraging, especially for imbalanced datasets. It suggests that for the anomalies it *does* identify (or scores highly), it does so with good precision.
    *   **Score Distribution:** Figure `autoencoder_anomaly_scores_distribution.png` visually confirms the findings. While there's a clear shift in the mean anomaly score for actual anomalies (red distribution shifted right), there's considerable overlap with the normal scores distribution, explaining why a single threshold struggles to capture all anomalies without increasing false positives. The anomalies the model *is* confident about (very high scores) are indeed distinct.
    *   **Limitations:** The current recall suggests that many injected anomalies are too subtle for the autoencoder to distinguish with high reconstruction error at the chosen threshold, or that the "normal" data itself has a wide range of variability that makes some anomalies look similar to normal extremes.

### Code Insights
*   **Modular Structure:** The Python code is organized into scripts (`prepare_data.py`, `train_forecasting.py`, `train_anomaly_detection.py`, `evaluate_models.py`) and utility/configuration files (`utils.py`, `config.py`). This promotes modularity and reusability.
*   **Data Augmentation Logic:** The `synth_day` function in `main.ipynb` (and presumably moved to `utils.py` or similar for the scripts) is a key component, allowing for the creation of a richer dataset. The strategy of using minute-by-minute statistics from a real day to generate new days with controlled noise and anomaly injection is a practical approach for limited initial data.
*   **Model Architectures:**
    *   **Forecasting (LSTM/TCN):** Both models use common best practices like Bidirectional LSTMs (for LSTM), causal/dilated convolutions (for TCN), Batch Normalization for stability, and Dropout for regularization. Huber loss was chosen for robustness to outliers in time series.
    *   **Autoencoder:** The architecture is a stacked Bidirectional LSTM encoder-decoder, designed to capture complex temporal patterns. Training it only on normal data is standard practice for this type of anomaly detection.
*   **Training and Evaluation:**
    *   Chronological splitting is correctly implemented, vital for time series.
    *   Use of `tf.data.Dataset` for efficient data loading pipeline in TensorFlow.
    *   Comprehensive callbacks (`EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`) are used to manage training effectively.
    *   The threshold optimization for the Autoencoder based on validation set F1-score is a data-driven approach to improve classification performance.
*   **Configuration Management:** `config.py` centralizes parameters, making it easy to experiment with different settings (sequence lengths, model hyperparameters, file paths).
*   **Reproducibility:** Saving scalers (`.joblib`) and models (`.keras`) allows for consistent preprocessing and model reuse.
*   **Challenges Addressed in Code:**
    *   Handling limited initial data via augmentation.
    *   Ensuring robust training through appropriate loss functions and regularization.
    *   Systematic evaluation with relevant metrics for both forecasting and anomaly detection.

### Figures and Tables of Results

**Table 1: Forecasting Model Performance on Test Set (Denormalized Metrics)**
| Model | Metric | Value     |
| :---- | :----- | :-------- |
| LSTM  | MSE    | [Value from plot19 e.g., 0.53] |
| LSTM  | MAE    | [Value from plot19 e.g., 0.52] |
| LSTM  | RMSE   | [Value from plot19 e.g., 0.73] |
| TCN   | MSE    | [Value from plot19 e.g., 0.36] |
| TCN   | MAE    | [Value from plot19 e.g., 0.40] |
| TCN   | RMSE   | [Value from plot19 e.g., 0.60] |
*Figure `forecasting_model_comparison.png` visually represents this data.*
`![Forecasting Model Comparison](images/forecasting_model_comparison.png)`

**Table 2: Autoencoder Anomaly Detection Performance on Test Set**
| Metric            | Value |
| :---------------- | :---- |
| Precision         | 0.9841 |
| Recall            | 0.2938 |
| F1 Score          | 0.4526 |
| AUC Score (ROC)   | 0.8297 |
| Average Precision (PR)| 0.9369 |
*(Source: `anomaly_metrics.csv`)*

**Table 3: Autoencoder Threshold Analysis (on Validation Set)**
| Threshold | Precision | Recall   | F1-Score | FP | FN  |
| :-------- | :-------- | :------- | :------- | :- | :-- |
| 0.2879    | 0.9784    | 0.3250   | 0.4879   | 6  | 565 |
| 0.3355    | 0.9865    | 0.2628   | 0.4151   | 3  | 617 |
| 0.4511    | 0.9820    | 0.1959   | 0.3267   | 3  | 673 |
| 0.6963    | 1.0000    | 0.1338   | 0.2360   | 0  | 725 |
| 0.8385    | 1.0000    | 0.1063   | 0.1922   | 0  | 748 |
| 1.8158    | 1.0000    | 0.0669   | 0.1254   | 0  | 781 |
| 2.5676    | 1.0000    | 0.0406   | 0.0781   | 0  | 803 |
| 2.7560    | 1.0000    | 0.0275   | 0.0535   | 0  | 814 |
| 3.0018    | 1.0000    | 0.0143   | 0.0283   | 0  | 825 |
| 3.2081    | 1.0000    | 0.0072   | 0.0142   | 0  | 831 |
*(Source: `autoencoder_threshold_analysis.csv`)*

**Key Visualizations:**

*   **Data Augmentation & EDA:**
    `![Augmented Transactions with Anomalies](images/augmented_transactions_with_anomalies.png)`
    `![Feature Distributions Original vs Augmented (numero_transazioni)](images/dist_numero_transazioni_orig_vs_aug.png)`
    `![Feature Correlation Matrix](images/feature_correlation_matrix.png)`
    `![Anomaly Distribution Across Splits](images/anomaly_distribution_splits.png)`

*   **Forecasting Model Results:**
    `![LSTM Forecast vs Actual](images/lstm_forecast_vs_actual.png)`
    `![TCN Forecast vs Actual](images/tcn_forecast_vs_actual.png)`
    `![Forecasting Model Training History](images/forecasting_training_history.png)`

*   **Autoencoder Anomaly Detection Results:**
    `![Autoencoder Training History](images/autoencoder_training_history.png)`
    `![Autoencoder Anomaly Scores Distribution](images/autoencoder_anomaly_scores_distribution.png)`
    `![Autoencoder Confusion Matrix](images/autoencoder_confusion_matrix.png)`
    `![Autoencoder ROC Curve](images/autoencoder_roc_curve.png)`
    `![Autoencoder Precision-Recall Curve](images/autoencoder_pr_curve.png)`

---

## [Section 5] Conclusions

### Take-away Points
This project successfully implemented an end-to-end anomaly detection pipeline for transactional time series data, starting from a single day of data and leveraging augmentation techniques to create a more comprehensive dataset. An LSTM-based Autoencoder was developed and trained to identify anomalies. While the Autoencoder demonstrated very high precision (0.984), meaning its identified anomalies are highly reliable, its recall was modest (0.294), indicating it missed a significant portion of the synthetically injected anomalies. The AUC-ROC (0.830) and AUC-PR (0.937) scores suggest good overall discriminative power. The threshold optimization process was crucial in balancing precision and recall. The forecasting models (LSTM and TCN) showed capability in learning normal patterns, with TCN slightly outperforming LSTM on basic regression metrics.

### Limitations and Problems Encountered
*   **Synthetic Anomalies:** The anomalies detected were synthetically injected. Their characteristics (magnitude, type) might not fully represent the diversity and subtlety of all real-world operational anomalies. The model's performance on true, unforeseen anomalies remains to be validated.
*   **Augmentation Basis:** The entire augmented week was based on the patterns of a single real day. If this day was not representative of all typical behaviors (e.g., special events, different days of the week having vastly different underlying patterns not captured by the minute-of-day stats), the model's understanding of "normal" might be skewed.
*   **Low Recall for Autoencoder:** Despite good precision, the Autoencoder missed many anomalies. This could be due to:
    *   The chosen threshold being too conservative to capture more subtle anomalies.
    *   Some injected "anomalies" being too similar to normal fluctuations after scaling and noise addition in the augmentation.
    *   The Autoencoder architecture needing further tuning or a different approach for more nuanced anomalies.
*   **Initial Data Outliers:** The original `tempo_max` contained extreme outliers, which were handled by focusing on `tempo_medio` and using clipping during augmentation, but initial data quality can always impact downstream modeling.
*   **Feature Set:** The analysis relied on four primary features. Other potentially relevant information (e.g., specific error codes, server resource metrics) was not available but could improve detection.
*   **Computational Resources:** Training deep learning models like LSTMs, TCNs, and Autoencoders can be computationally intensive, especially with longer sequences or more complex architectures.

### Future Work
*   **Testing on Real, Diverse Data:** The most critical next step is to evaluate the system on a larger, real-world dataset spanning multiple days/weeks, including naturally occurring (and expertly verified) anomalies.
*   **Explore Other Anomaly Detection Algorithms:**
    *   Implement and compare with classical statistical methods (e.g., ARIMA-based residual analysis).
    *   Test other unsupervised machine learning algorithms like Isolation Forest, One-Class SVM, or clustering-based approaches (e.g., DBSCAN on reconstruction errors or feature embeddings).
*   **Refine Autoencoder Architecture/Training:**
    *   Experiment with different Autoencoder architectures (e.g., Variational Autoencoders - VAEs, different layer types or sizes).
    *   Explore more sophisticated loss functions or training strategies for anomaly detection.
*   **Advanced Feature Engineering:** Incorporate more domain-specific features or interaction terms if available.
*   **Ensemble Methods:** Combine predictions from multiple models (e.g., forecasting error-based approach + Autoencoder scores) to potentially improve robustness and performance.
*   **Online Learning/Adaptation:** Develop mechanisms for the models to adapt to evolving "normal" patterns (concept drift) over time.
*   **Explainability (XAI):** Integrate techniques like SHAP or LIME to understand why specific data points are flagged as anomalous by the Autoencoder, increasing trust and actionability.
*   **Hyperparameter Optimization:** Conduct systematic hyperparameter tuning (e.g., using KerasTuner or Optuna) for all models to potentially find better configurations.
```

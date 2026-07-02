# Restaurant Demand Forecasting with Databricks 🍽️

Machine learning system that forecasts daily visitor demand for individual restaurants and attaches P10 to P90 confidence ranges to every prediction, built end to end on Databricks with time series cross validation and MLflow experiment tracking.

---

## Overview 📍

- **Input:** Per store daily visit history plus calendar and store features (day of week, holidays, genre, area)
- **Output:** Daily visitor forecast per restaurant, with a P10 / P50 / P90 range instead of a single number
- **Final Model:** Global LightGBM (gradient boosting) using lag, rolling and calendar features
- **Final Performance:** RMSLE 0.594 (±0.069) across 3 cross validation folds, versus a 0.668 baseline
- **Pipeline:** Delta ingestion → EDA → feature engineering → baseline → model comparison → cross validation and quantile calibration
- **Platform:** Databricks, Delta tables, MLflow

---

## Dataset 📊

The dataset comes from the Recruit Restaurant Visitor Forecasting competition on Kaggle, which contains daily visitor counts for roughly 800 Japanese restaurants between 2016 and April 2017. The counts originate from an actual point of sale / register system, and the data is accompanied by a reservation platform, store metadata (genre and area) and a calendar with public holiday flags. Of the eight files available, three were used for this version: the daily visit counts (the forecasting target), the store information (used for cohorts and features) and the calendar (day of week and holidays). This was an intentional choice to keep the first build focused on a complete forecasting pipeline rather than maximising inputs. The reservation files were deliberately left out as the highest value addition for a later iteration, since reservations act as a leading indicator of visits. One important characteristic of the data is that days on which a store was closed are omitted, which creates irregular gaps that had to be handled explicitly during modelling.

---

## Model Selection 🔍

Model selection was driven by the need for a single model that forecasts every store accurately, including the many stores with short history. A seasonal naive baseline (predicting each day from the same weekday one week earlier) was established first, reaching an RMSLE of 0.668, and any real model had to beat this to be worth keeping. A classical AutoETS model was then tested as a per series statistical approach. It beat the baseline (RMSLE 0.622) but could only be fitted on stores with enough history and was sensitive to the closed day gaps, so it could not cover the full set of restaurants. A global LightGBM model, trained across all stores on engineered lag, rolling and calendar features, achieved the best result (RMSLE 0.537 on a single window) while covering every store, including the short history ones that the classical model cannot fit at all. The target was modelled on a log scale because visitor counts are heavily right skewed, which stabilises training and shifts the objective toward proportional error. To avoid relying on a single lucky test window, the final model was validated with rolling time series cross validation across three folds, giving a more honest estimate of RMSLE 0.594 (±0.069). The small spread across folds indicates stable performance over time, with the hardest fold falling over the New Year holiday period.

---

## Uncertainty and Calibration 🎯

Rather than producing only a point forecast, the model was extended with quantile regression to output a P10 to P90 range for each store and day, giving a low, middle and high estimate that can be used for staffing or stock decisions. The reliability of these ranges was then measured directly by checking what fraction of actual values fell inside the P10 to P90 band, which should sit around 80 percent. Observed coverage was 93.4 percent, meaning the bands are currently too wide and the model is underconfident rather than overconfident. This is the safer direction to err, since the intervals never project false precision, and the write up notes how they would be tightened using narrower target quantiles or conformal calibration.

---

## Assumptions and Limitations 🚧

The model assumes that recent demand patterns per store, together with day of week, holiday and store level features, sufficiently capture future visitor variation. Because the dataset is a historical snapshot from 2016 to 2017, it is a demonstration of method rather than a current market model, and a live system would be affected by data drift and require scheduled retraining. Closed days were filled as zero so the model could learn each store's closure rhythm, but scoring was restricted to genuinely open days, since RMSLE penalises any positive prediction on a truly closed day extremely harshly. Evaluation also assumes a short horizon setup where recent actuals are available, so true multi week ahead forecasting would be a harder extension. Predictions should be read as indicative estimates rather than precise counts.

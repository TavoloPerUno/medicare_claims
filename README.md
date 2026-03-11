# Anomaly Detection in Medicare Claims

Detecting anomalous billing patterns and predicting overcharge ratios in Medicare Part-B claims using unsupervised outlier detection and supervised machine learning. Built on 2015 CMS physician utilization data for Illinois (~802K claims across 30K+ physicians).

## Key Findings

- **XGBoost with logit-transformed target** achieves the best predictive performance (R² = 0.69) for overcharge ratio prediction.
- Claim amount, organization size, number of services, department, and graduation year are the strongest predictive features.
- Similarity-based outlier detection using HCPCS code co-occurrence successfully identifies physicians with atypical service code combinations within their department, with anomaly scores following a Zipf distribution.
- Higher submitted charge amounts correlate with higher overcharge ratios (1 - payment/charge).

## Methods

### Unsupervised: Service Code Anomaly Detection

Identifies physicians with unusual billing patterns by analyzing HCPCS service code co-occurrence. For each physician, a "code cluster diameter" is computed using CountVectorizer text similarity on their service code combinations. Department-specific percentile cutoffs (95th and 99th) flag physicians whose code diversity deviates significantly from peers in the same specialty.

### Supervised: Overcharge Ratio Prediction

Predicts the overcharge ratio (1 - Medicare payment / submitted charge) using 197 features derived from claims data, physician demographics, patient experience surveys, and clinical performance scores. Models are evaluated with 5-fold cross-validation on a 67/33 train-test split.

| Model | R² Score | MSE |
|---|---|---|
| XGBoost (Logit Transform) | 0.686 | 0.012 |
| XGBoost | 0.668 | 0.012 |
| Random Forest | 0.543 | 0.017 |
| Logit Regression | 0.307 | 0.026 |
| Linear Regression | 0.251 | 0.028 |
| Deep Neural Network (TensorFlow) | 0.210 | 0.029 |

## Data

This study uses publicly available CMS Medicare Part-B data for 2015, filtered to Illinois providers:

- **[Medicare Provider Utilization and Payment Data](https://www.cms.gov/Research-Statistics-Data-and-Systems/Statistics-Trends-and-Reports/Medicare-Provider-Charge-Data/Physician-and-Other-Supplier)** -- physician-level claims with HCPCS codes, charges, and payments
- **[Physician Compare National Downloadable File](https://data.cms.gov/provider-data/)** -- physician demographics, specialties, and practice information
- **Physician Compare Performance Scores** -- individual clinical quality measures
- **Physician Compare Patient Experience** -- group-level patient survey results

Data files are not included in this repository due to size. Download from the CMS links above and place them in a `data/` directory.

## Project Structure

```
medicare_claims/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── report.ipynb            # Main analysis notebook with visualizations
├── figures/                     # Output plots from notebook analysis
│   └── output_*.png
└── src/
    └── medicare_analysis/
        ├── __init__.py
        ├── analyse.py           # CLI entry point for running models
        ├── models.py            # Model training (Linear, RF, XGBoost, DNN)
        ├── logit_regression.py  # Logit-transformed regression wrappers
        └── preprocess_files.py  # Data loading and feature engineering
```

## Getting Started

### Installation

```bash
git clone https://github.com/TavoloPerUno/medicare_claims.git
cd medicare_claims
pip install -r requirements.txt
```

### Running the Analysis

The main analysis and results are in the Jupyter notebook:

```bash
jupyter notebook notebooks/report.ipynb
```

To train individual models from the command line:

```bash
cd src/medicare_analysis
python analyse.py lreg      # Linear regression
python analyse.py breg      # Logit regression
python analyse.py rforest   # Random forest
python analyse.py xgboost   # XGBoost
python analyse.py nnet      # Deep neural network
python analyse.py cluster   # Unsupervised anomaly detection
```

## References

1. Berwick, D. M., & Hackbarth, A. D. (2012). Eliminating Waste in US Health Care. *JAMA*, 307(14), 1513-1516. [doi:10.1001/jama.2012.362](https://doi.org/10.1001/jama.2012.362)

2. Bauder, R. A., & Khoshgoftaar, T. M. (2017). Medicare Fraud Detection Using Machine Learning Methods. *2017 16th IEEE International Conference on Machine Learning and Applications (ICMLA)*, 858-865. [doi:10.1109/ICMLA.2017.00-48](https://doi.org/10.1109/ICMLA.2017.00-48)

3. Herland, M., Khoshgoftaar, T. M., & Bauder, R. A. (2018). Big Data fraud detection using multiple medicare data sources. *Journal of Big Data*, 5, 29. [doi:10.1186/s40537-018-0138-3](https://doi.org/10.1186/s40537-018-0138-3)

4. Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly detection: A survey. *ACM Computing Surveys*, 41(3), 1-58. [doi:10.1145/1541880.1541882](https://doi.org/10.1145/1541880.1541882)

5. Joudaki, H., Rashidian, A., Minaei-Bidgoli, B., Mahmoodi, M., Geraili, B., Nasiri, M., & Arab, M. (2015). Using Data Mining to Detect Health Care Fraud and Abuse: A Review of Literature. *Global Journal of Health Science*, 7(1), 194-202. [doi:10.5539/gjhs.v7n1p194](https://doi.org/10.5539/gjhs.v7n1p194)

6. Centers for Medicare & Medicaid Services. [Medicare Provider Utilization and Payment Data: Physician and Other Supplier](https://www.cms.gov/Research-Statistics-Data-and-Systems/Statistics-Trends-and-Reports/Medicare-Provider-Charge-Data/Physician-and-Other-Supplier).

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

"""
<a href="https://colab.research.google.com/github/Xquisite-design/enterprise-presales-knowledge-orchestration/blob/main/notebooks/01_Data_Loading_Quality_Cleaning.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Part 1 of 5 : Data Loading, Quality Assessment and Cleaning

Loads the Curated Enterprise Presales Research Corpus (CEPRC), assesses data quality and applies cleaning.
The cleaned dataset is saved to Drive for the remaining notebooks.

**Next:** `02_Exploratory_Data_Analysis.ipynb`

The Curated Enterprise Presales Research Corpus (CEPRC) consists of 900 annotated enterprise presales question–answer pairs represented by 29 variables. The dataset contains a combination of 23 categorical variables, 3 numerical variables, and 3 placeholder variables intended for future enhancement. Overall, the dataset is compact (approximately 204 KB) and suitable for exploratory analysis, statistical modelling, and machine learning

Since our dataset is an annotated enterprise presales corpus (around 900 observations with mostly categorical variables), I have elected to do EDA which is research-question driven rather than simply showing histograms and boxplots.

We will do the analysis in this order:

1. Dataset loading and inspection
2. Data quality assessment
3. Data cleaning log
4. EDA (Univariate and Bivariate EDA (aligned to each Research Question)
5. Inferential Statistics (RQ specific)
6. Feature engineering
7. Baseline ML models
8. Preliminary evaluation
9. GitHub documentation

#PART-A : EDA

## 1. Dataset loading and inspection
"""


def mount_drive(mount_point: str = "/content/drive") -> None:
    """
    Mount Google Drive storage for accessing datasets and persisting project artifacts.
    """
    from google.colab import drive

    drive.mount(mount_point)


if __name__ == "__main__":
    mount_drive()

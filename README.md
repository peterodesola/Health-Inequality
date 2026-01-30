# End‑to‑end Health Inequality Analytics with Python & Streamlit

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset And Features](#dataset-and-features)
- [Problem Statement And Key Questions](#problem-statement-and-key-questions)
  - [Problem Statement](#problem-statement)
  - [Key Questions](#key-questions)
- [Data Cleaning & Preprocessing](#data-cleaning-&-preprocessing)
  - [Encoding & Header Fixes](#encoding-&-header-fixes)
  - [Handling Missing Values](#handling-missing-values)
  - [Type Conversion](#type-conversion)
  - [Feature Engineering](#feature-engineering)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-(eda))
  - [1. Distribution & KPIs](#1-distribution-&-kpis)
    - [Descriptive Statistics (Count, Mean, STD, Min)](#descriptive-statistics-(count-mean-std-min))
    - [Descriptive Statistics (Quartiles And Max)](#descriptive-statistics-(quartiles-and-max))
    - [GII Distributions](#gii-distributions)
  - [2. Relationships Between GII And Key Variables](#2-relationships-between-gii-and-key-variables)
    - [GII vs Maternal_Mortality](#gii-vs-maternal_mortality)
    - [GII vs Adolescent_Birth_Rate](#gii-vs-adolescent_birth_rate)
    - [GII vs Seats_Parliamentt-Held-By-Women](#gii-vs-seats_parliamenttheld-by-women)
    - [GII vs Edu_Gap](#gii-vs-edu_gap)
  - [3. GII By Human Development Group (box plot)](#3-gii-by-human-development-group-(box-plot))
  - [4. Correlation Heatmap](#4-correlation-heatmap)
- [Predictive Modeling – Random Forest After Feature Engineering](#predictive-modeling--random-forest-after-feature-engineering)
  - [1. Data Preparation For Modeling](#1-data-preparation-for-modeling)
  - [2. Baseline Random Forest Performance (5-fold CV)](#2-baseline-random-forest-performance-(5-fold-cv))
- [Results And Discussion](#results-and-discussion)
  - [Overall Distribution Of Gender Inequality](#overall-distribution-of-gender-inequality)
  - [Relationships With Health Representation And Gender Gaps](#relationships-with-health-representation-and-gender-gaps)
  - [Predictive Modeling Performance](#predictive-modeling-performance)
- [Conclusion](#conclusion)


## Project overview

Despite global progress, gender‑based health and socioeconomic inequalities remain substantial across countries. Policymakers often lack intuitive, data‑driven tools to explore where inequalities are largest, which structural factors drive them, and how potential policy changes might shift outcomes.

In this project, I used a country‑level Gender Inequality Index (GII) dataset and built an end‑to‑end analytics pipeline and interactive Streamlit web app to help answer:

- How does gender inequality vary across countries and human development groups?
- Which health and socioeconomic indicators are most strongly associated with higher GII values?
- Given hypothetical improvements in these drivers, how might a country’s GII change?

---

## Dataset and features

I worked with a cross‑country dataset of gender inequality indicators (~150+ countries) containing:

- **HDI rank** – Human Development Index rank (1 = highest development).
- **Country** – Country name.
- **HUMAN DEVELOPMENT** – Development level (e.g. `VERY HIGH`, `HIGH`, `MEDIUM`).
- **GII VALUE** – Gender Inequality Index; composite measure (0 = equality, 1 = highest inequality).
- **GII RANK** – Rank of the country by GII.
- **Maternal_mortality** – Maternal deaths per 100 000 live births.
- **Adolescent_birth_rate** – Births per 1 000 girls aged 15–19.
- **Seats_parliamentt(% held by women)** – Percentage of parliamentary seats held by women.
- **F_secondary_educ / M_secondary_educ** – % of women/men with at least secondary education.
- **F_Labour_force / M_Labour_force** – Female/male labour force participation rate (%).

Engineered features:

- **Edu_gap** = `M_secondary_educ – F_secondary_educ` (education gap).
- **Labour_gap** = `M_Labour_force – F_Labour_force` (labour force gap).

---

## Problem statement & key questions

### Problem statement

Gender inequality is multi‑dimensional, spanning health outcomes, education, political representation and labour markets. Country‑level indices like GII summarise these dimensions but can be hard to interpret and act upon without deeper analysis and accessible tools.  

The goal of this project is to turn a static GII dataset into an interactive, insight‑driven application that supports monitoring, explanation and scenario analysis for gender inequality across countries.

### Key questions

From a data and policy perspective, I focused on:

1. Which development groups show the highest and lowest GII values?
2. How strongly are GII values associated with maternal mortality and adolescent birth rates?
3. How do education and labour‑force gaps relate to gender inequality?
4. Can we build a reasonably accurate model to predict GII from a small set of interpretable indicators?
5. How can we package these insights into a clean, interactive web app for non‑technical users?

---

## Data cleaning & preprocessing

Using Python (`pandas`, `numpy`), I applied:

### Encoding & header fixes

- Loaded the CSV with `encoding="utf-8-sig"` to strip BOM artifacts from the `HDI rank` header (which appeared as `ï»¿HDI rank` initially). This ensures the column is correctly read as `HDI rank`.

### Handling missing values

- Replaced placeholder values like `..` with `NaN`.
- For EDA:
  - Retained as many rows as possible, using functions that handle missing values gracefully.
- For modeling:
  - Dropped rows with missing values in the target (`GII VALUE`) or required features.

### Type conversion

- Ensured all numeric columns were correctly typed:
  - `HDI rank`, `GII VALUE`, `GII RANK`
  - `Maternal_mortality`, `Adolescent_birth_rate`
  - `Seats_parliamentt(% held by women)`
  - `F_secondary_educ`, `M_secondary_educ`
  - `F_Labour_force`, `M_Labour_force`
- Used `pd.to_numeric(..., errors="coerce")` to convert and set non‑numeric values to `NaN`.

### Feature engineering

- **Edu_gap** = `M_secondary_educ – F_secondary_educ`
- **Labour_gap** = `M_Labour_force – F_Labour_force`

These derived features make gender gaps explicit and easier to analyse and model.

---

## Exploratory Data Analysis (EDA)

I used `seaborn`, `matplotlib`, and Plotly (via Streamlit) to explore the data visually.

### 1. Distribution & KPIs

**What I did**

- Computed summary statistics (mean, median, quartiles, min/max) for `GII VALUE` and other numeric indicators.
## Descriptive statistics (count, mean, std, min)

| Variable                          | count |    mean    |    std     |   min  |
|-----------------------------------|:-----:|:----------:|:----------:|:------:|
| HDI rank                          | 195.0 | 97.815385  | 56.467551  |  1.000 |
| GII VALUE                         | 170.0 | 0.344376   | 0.197105   |  0.013 |
| GII RANK                          | 170.0 | 85.376471  | 49.210206  |  1.000 |
| Maternal_mortality                | 181.0 | 143.839779 | 197.596395 |  2.000 |
| Adolescent_birth_rate             | 195.0 | 44.597949  | 38.422479  |  1.600 |
| Seats_parliamentt(% held by women)| 193.0 | 24.701554  | 12.404319  |  0.000 |
| F_secondary_educ                  | 177.0 | 62.706780  | 29.528487  |  6.400 |
| M_secondary_educ                  | 177.0 | 67.068362  | 26.450033  | 13.000 |
| F_Labour_force                    | 180.0 | 50.224444  | 15.905491  |  6.000 |
| M_Labour_force                    | 180.0 | 69.863333  |  9.012734  | 43.900 |
| Edu_gap                           | 177.0 | 4.361582   | 7.163829   | -25.400|
| Labour_gap                        | 180.0 | 19.638889  | 13.895899  | -1.600 |

## Descriptive statistics (quartiles and max)

| Variable                          |   25%   |   50%  |    75%    |  max   |
|-----------------------------------|:-------:|:------:|:---------:|:------:|
| HDI rank                          | 49.5000 | 97.000 | 146.00000 | 195.00 |
| GII VALUE                         | 0.1775  | 0.363  | 0.50575   | 0.82   |
| GII RANK                          | 43.2500 | 85.500 | 127.75000 | 170.00 |
| Maternal_mortality                | 12.0000 | 52.000 | 183.00000 | 917.00 |
| Adolescent_birth_rate             | 10.7500 | 36.200 | 64.20000  | 170.50 |
| Seats_parliamentt(% held by women)| 16.5000 | 23.600 | 33.60000  | 55.70  |
| F_secondary_educ                  | 37.7000 | 69.900 | 90.20000  | 100.00 |
| M_secondary_educ                  | 44.8000 | 71.200 | 92.50000  | 100.00 |
| F_Labour_force                    | 42.4000 | 52.150 | 60.47500  | 83.10  |
| M_Labour_force                    | 65.0500 | 69.350 | 75.52500  | 95.50  |
| Edu_gap                           | 0.0000  | 3.200  | 8.30000   | 28.80  |
| Labour_gap                        | 9.3500  | 15.600 | 25.85000  | 61.60  |

## GII Distributions
<img width="1638" height="602" alt="GII Distribution" src="https://github.com/user-attachments/assets/bd16ceaf-0e72-4dd2-aa8c-4520dbc76263" />

The left plot in the images above shows the overall distribution of GII VALUE, and the right plot summarises it with a boxplot.

### Histogram + KDE (left)

- Values range roughly from just above 0 up to around 0.8, so the dataset covers countries from very low to quite high gender inequality.  
- The density curve suggests a *skewed distribution:* many countries sit in the mid‑range (around 0.3–0.5), with fewer at the extremes near 0 or 0.8.  
- There is a noticeable mass at low GII values (high‑equality countries), but the tail extends towards higher values, indicating a subset of countries with substantially worse gender inequality.

### Boxplot (right)

- The median GII is around 0.35–0.4, matching the summary table where the 50th percentile is 0.363.
- The interquartile range (IQR) stretches roughly from about 0.18 to about 0.50, aligning with the 25th and 75th percentiles (0.1775 and 0.50575) in the stats.
- The whiskers reach close to the minimum (~0.013) and maximum (~0.82) GII values, capturing the full spread of inequality across countries.  
- Any points beyond the whiskers would be countries that are outliers in terms of unusually low or high gender inequality relative to the rest of the world.

## Relationship between GII and key variables using scatter plot
<img width="956" height="756" alt="GII Scatterplots relationship" src="https://github.com/user-attachments/assets/9aba8be3-b0cf-4300-bdee-c86083224d97" />

### 1. GII vs Maternal_mortality

- **Overall pattern:** Clear positive relationship — as maternal mortality increases, GII tends to rise.
- **Low mortality range:** Countries with very low maternal mortality (close to 0–50 deaths per 100 000 live births) mostly have low GII values, especially among VERY HIGH and HIGH human development groups.
- **High mortality range:** Medium and low development countries with very high maternal mortality show some of the highest GII values in the dataset.
- **Interpretation:** Poor maternal health outcomes are strongly associated with greater gender inequality.

---

### 2. GII vs Adolescent_birth_rate

- **Overall pattern:** Strong positive association — higher adolescent birth rates coincide with higher GII values.
- **Lower birth rates:** Countries with low adolescent birth rates (roughly <20 births per 1 000 girls 15–19) cluster at lower GII values, again dominated by higher‑development countries.
- **Higher birth rates:** Medium and low development countries with adolescent birth rates above ~60–80 often show mid‑to‑high GII levels.
- **Interpretation:** Early childbearing is closely linked to gender inequality, reflecting constraints on girls’ education and autonomy.

---

### 3. GII vs Seats_parliamentt(% held by women)

- **Overall pattern:** Negative relationship — as the percentage of seats held by women in parliament increases, GII tends to decrease.
- **Low representation:** Countries with very low women’s representation (0–10%) often show moderate to high GII values, particularly in medium and low development categories.
- **Higher representation:** Countries with 30–50% women in parliament are more frequently found at lower GII levels, especially among very‑high and high development groups.
- **Interpretation:** Greater political representation of women is associated with more gender‑equal societies.

---

### 4. GII vs Edu_gap

- **Edu_gap definition:** Difference between male and female secondary education rates (`M_secondary_educ – F_secondary_educ`).
- **Overall pattern:** Mostly positive association — larger education gaps tend to align with higher GII, though the relationship is more dispersed than for mortality and adolescent births.
- **Small or negative gaps:** When the education gap is close to zero (or slightly negative, where women’s schooling matches or exceeds men’s), many countries have relatively low GII, often in the VERY HIGH and HIGH development groups.
- **Large gaps:** As Edu_gap increases (men much more educated than women), GII values generally move into the medium and high ranges.
- **Interpretation:** Unequal access to education between men and women is a structural driver of gender inequality.


## Interpretation of GII by Human Development group (box plot)

<img width="818" height="555" alt="GII by human development" src="https://github.com/user-attachments/assets/b72ad6b3-095b-4878-9975-78f9e05c18bd" />

### Overall pattern

- The box plot compares **GII VALUE** across four **HUMAN DEVELOPMENT** categories: `VERY HIGH`, `HIGH`, `MEDIUM`, and `LOW`. 
- There is a clear upward shift in GII as development level decreases: from lowest medians in `VERY HIGH` to highest in `LOW`.

---

### VERY HIGH human development

- **Median GII** is low (around 0.1), with the box mostly between roughly 0.05 and 0.20.  
- The lower whisker is close to 0, and the upper whisker extends to just below 0.4.  
- Interpretation: most very‑high‑development countries experience comparatively low gender inequality, with relatively limited variation between them.

---

### HIGH human development

- **Median GII** is higher (around 0.35), and the IQR is wider than for `VERY HIGH`.  
- The distribution ranges from about 0.15 up to >0.6, with some higher‑inequality outliers.  
- Interpretation: high‑development countries show moderate inequality on average, but with substantial differences between individual countries.

---

### MEDIUM human development

- **Median GII** is around the 0.5 mark, clearly higher than in the previous two groups.  
- The box (middle 50% of countries) lies roughly between 0.45 and 0.55, with whiskers and a few outliers reaching above 0.6–0.7.  
- Interpretation: medium‑development countries generally face significant gender inequality, with many clustered in a high‑GII band.

---

### LOW human development

- **Median GII** is the highest of all groups, close to 0.6.  
- The IQR sits roughly between ~0.57 and ~0.63, with whiskers and outliers extending towards ~0.7–0.8.  
- Interpretation: low‑development countries systematically experience the greatest gender inequality, and even the “best” performers in this group still have relatively high GII values.

---
## Correlation heatmap  

<img width="1053" height="763" alt="confusion matrix" src="https://github.com/user-attachments/assets/b2eda711-ba26-4d97-8d94-97762a182907" />

- **GII VALUE vs health outcomes**
  - GII is **strongly positively correlated** with both `Maternal_mortality` (~0.75) and `Adolescent_birth_rate` (~0.81).
  - Interpretation: countries with higher maternal deaths and higher teenage birth rates tend to have **much higher gender inequality**.

- **GII VALUE vs women’s education and empowerment**
  - GII is **strongly negatively correlated** with `F_secondary_educ` (~−0.81) and `M_secondary_educ` (~−0.78).
  - It is **moderately negatively correlated** with `Seats_parliamentt(% held by women)` (~−0.42).
  - Interpretation: higher female (and overall) education and greater women’s representation in parliament are associated with **lower gender inequality**.

- **GII VALUE vs labour market**
  - Correlations with `F_Labour_force` and `M_Labour_force` are weak (near zero), indicating that **overall participation rates alone** are not strong predictors of GII.
  - However, `Labour_gap` (male–female gap) has a small positive correlation with GII (~0.17), while `F_Labour_force` is strongly **negatively correlated** with `Labour_gap` (~−0.83), meaning larger gaps arise where women’s participation is especially low.

- **GII VALUE vs gaps**
  - `Edu_gap` shows a **moderate positive correlation** with GII (~0.45): as the male–female education gap widens, gender inequality rises.
  - Taken together, the matrix highlights **reproductive health indicators and gender gaps in education** as the strongest correlates of inequality in this dataset.
 

  ## Predictive modeling – Random Forest after feature engineering

After EDA, I trained a Random Forest regression model to predict **GII VALUE** from the most relevant health and gender‑gap indicators.

### 1. Data preparation for modeling

Before training, I refined the dataset to make the learning problem as clean as possible:

- **Target selection**

  - `y = df_model["GII VALUE"]` – the continuous GII score to be predicted.

- **Outlier handling**

  - Clipped extreme values (1st–99th percentile) for the two most skewed drivers:
    - `Maternal_mortality`
    - `Adolescent_birth_rate`  
  - This limits the influence of a few extreme countries on the model while retaining most of the data.

- **Feature transformation**

  - Applied log‑transforms to stabilise variance and reduce skew:
    - `log_Maternal_mortality = log1p(Maternal_mortality)`
    - `log_Adolescent_birth_rate = log1p(Adolescent_birth_rate)`

- **Feature selection**

  - Focused on a compact, interpretable set of predictors:

    ```python
    features = [
        "log_Maternal_mortality", "log_Adolescent_birth_rate",
        "Seats_parliamentt(% held by women)",
        "F_secondary_educ", "M_secondary_educ",
        "F_Labour_force", "M_Labour_force",
        "Edu_gap", "Labour_gap"
    ]
    ```

  - Dropped rows with missing values in any of these features to ensure complete cases for modeling.

This produced design matrices:

- `X = df_model[features]`
- `y = df_model["GII VALUE"]`

ready for baseline evaluation and hyperparameter tuning.


### 2. Baseline Random Forest performance (5‑fold CV)

I first trained a **baseline RandomForestRegressor** with 500 trees and default depth settings:

python
rf_base = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    max_depth=None
)
scores_base = cross_val_score(rf_base, X, y, cv=5, scoring="r2")


## Results and discussion

### Overall distribution of gender inequality

The distribution of **GII VALUE** spans from very low values close to 0 up to about 0.8, with a median around 0.36 and an interquartile range of roughly 0.18–0.51. This indicates that, while a subset of countries achieves relatively low gender inequality, a large share falls in the mid‑to‑high range, and a non‑trivial minority experiences very high inequality. The box plot by human development group shows a clear gradient: **VERY HIGH** HDI countries cluster at low GII, **HIGH** countries at moderate levels, and **MEDIUM/LOW** groups at the highest levels, with medians stepping up almost monotonically across categories. This visually reinforces the strong link between overall human development and gender equality.

### Relationships with health, representation and gender gaps

Bivariate scatterplots reveal that **reproductive health variables are strongly associated with GII**. As maternal mortality and adolescent birth rates increase, GII rises sharply, particularly among medium‑ and low‑development countries. Conversely, higher percentages of parliamentary seats held by women are associated with lower GII, suggesting that political representation and empowerment are important correlates of more equal societies.

Education and labour‑force participation add another dimension. Countries where women’s secondary education and labour participation approach or exceed men’s tend to have lower GII values, while those with large **education (Edu_gap)** and **labour (Labour_gap)** gaps exhibit systematically higher inequality. The correlation heatmap quantifies these patterns: GII correlates positively with maternal mortality and adolescent births (≈0.75–0.81) and with Edu_gap (≈0.45), and negatively with female secondary education (≈−0.81) and women’s representation in parliament (≈−0.42). These magnitudes highlight that **structural gaps in health, schooling and political power** are central drivers of gender inequality in this dataset.

The global choropleth map makes the geography of inequality explicit. Countries in Western/Northern Europe, parts of North America and Oceania are shaded green (low GII), while many states in Sub‑Saharan Africa, South Asia and parts of the Middle East appear in yellow to red tones, reflecting substantially higher inequality. This aligns closely with the human development gradient and emphasises that gender inequality remains most acute where broader socioeconomic development is also constrained.

### Predictive modeling performance

On the predictive side, I experimented with **Random Forest regression** to estimate GII from a compact, interpretable feature set (reproductive health indicators, representation, education and labour gaps), with log‑transformations and outlier clipping for the most skewed variables. Despite this feature engineering, 5‑fold cross‑validation shows that the current models have **limited out‑of‑sample accuracy**: the baseline Random Forest achieves a mean R² of about −0.13 (std ≈ 0.85), and hyperparameter tuning via RandomizedSearchCV yields a best R² around −0.43, indicating no real improvement and some overfitting.

These negative R² scores mean that, on held‑out folds, the models perform worse than a naive predictor that always outputs the global mean GII. This is not unexpected: the dataset is relatively small (≈150–180 observations), there is substantial noise and heterogeneity across countries, and many relevant determinants of gender inequality (laws, norms, violence, social protection) are not explicitly captured in the available columns. As a result, the trained `gii_model_tuned.pkl` is best interpreted as a **scenario‑exploration tool** rather than a high‑precision forecasting engine.

Nevertheless, the modeling exercise is valuable. First, feature importance and correlations consistently point to the same set of key drivers (reproductive health, girls’ education, women’s political and economic participation), adding quantitative weight to qualitative policy narratives. Second, integrating the model into the Streamlit app allows policymakers and stakeholders to see how hypothetical improvements in these indicators might affect GII, even if approximate, which can support more engaged and evidence‑informed discussion.

<img width="1902" height="853" alt="gui" src="https://github.com/user-attachments/assets/30e28556-76d1-4425-9f0c-4e5ca46b9c81" />


## Conclusion

This project demonstrates how a global Gender Inequality Index dataset can be transformed from a static table into a **full analytical workflow and interactive decision‑support tool**. Starting from raw CSV data with encoding issues and missing values, I performed systematic cleaning, engineered gender‑gap features, and carried out a rich exploratory analysis that connected GII to reproductive health, education, labour markets and political representation. Visuals such as distribution plots, scatterplots, box plots, correlation matrices and a global choropleth map collectively reveal a clear story: gender inequality is lowest in very‑high‑development countries and highest where maternal mortality, adolescent birth rates and gender gaps in education and work remain large.

Although Random Forest regression models did not achieve strong predictive performance in cross‑validation, the exercise highlighted the **limits of modeling with small, noisy country‑level samples** and underscored the importance of additional covariates for robust prediction. The tuned model has been integrated into a Streamlit application as a **scenario tool**, allowing users to tweak key indicators and observe indicative changes in GII, while being transparent about uncertainty.



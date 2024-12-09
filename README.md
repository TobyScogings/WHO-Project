# We are team supermodellers!
This is the WHO project repo for team Supermodellers.

## Goal
Using the WHO Life Expectancy data, produce a function which takes in relevant population statistics and:
* Produce a **full model** to predict life expactancy using elaborate data (include sensitive information with user consent).
* The **full model** should produce an **RMSE lower than 2** compared to the competitor team's baseline.
* Produce a **minimalistic model** to predict life expectancy using minimal data (excludes sensitive information to remove potential biases).
* The **minimalistic model** should produce an **RMSE lower than 2.3** compared to the competitor team's baseline.

## Dataset
See `Life Expectancy Data.csv` and metadata info at `WHO Dataset - MetaData.ipynb`.

## Streamlit app
Give the [Life Expectancy Prediction app](https://supermodellers.streamlit.app/) a go!

## Tech-stack:
* `python`.
* `numpy`.
* `matplotlib`
* `seaborn`.
* `streamlit`.
* `scikit-learn`.
* `statsmodels`.


## Methodology
* Team experiments on EDA, feature-engineering, and finding the optimal full and minimalistic models.
* Discuss, compare results and decide on the final 2 models to use.
* Prepare the interactive functions for predictions.
* Present our work to technical and non-technical stakeholders.
* Submit the deliverables.

## Outcome
We managed to achieve our goals and produced the following final models:
* Full model RMSE on training data: `1.2`, R-Squared: `0.984`.
* Minimalistic model RMSE on training data: `2.18`, R-Squared: `0.946`. 

Please refer to [supermodellers project 1.ipynb](https://github.com/viviensiu/supermodellers/blob/main/Supermodellers%20Project%201.ipynb) for details on EDA, feature-engineering, and training the full and minimalistic models.

## To reproduce the Streamlit app locally
* Packages required (extra installation needed):
  * `streamlit`.
  * `scikit-learn`. 
* Files required within same folder (e.g. "supermodellers" folder):
  * `Life Expectancy Data.csv`: Contains training data to fit `StandardScaler` object for scaling.
  * `interactive_function.py`: Python script that makes up the Streamlit application.
  * `requirements.txt`: for setting up virtual environment on Streamlit Cloud, optional for local hosting.
  * `metadata.csv`: Required by `interactive_function.py`.
  * `app_header_image.png`: Required by `interactive_function.py`.
* In command prompt/terminal, navigate to the folder containing all the above files and run:
  `streamlit run interactive_function.py`   


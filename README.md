# We are team supermodellers!
This is the WHO project repo for team Supermodellers.

# Streamlit app
Give the [Life Expectancy Prediction app](https://supermodellers.streamlit.app/) a go!

# To reproduce the Streamlit app locally
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


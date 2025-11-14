# kcdi-streamlit-app

Web application for analysing the **Knowledge Contribution Diversity Index (KCDI)** from Scopus data.  
Built with Python and Streamlit.

## KCDI App for Epistemic Justice Analysis

### Description

This interactive web app computes the **Knowledge Contribution Diversity Index (KCDI)** to assess
epistemic justice in scholarly production. It allows users to upload a Scopus file (`.csv` or `.xlsx`)
and visualise diversity patterns in terms of gender and geography.

### Main features

- **KCDI analysis**: calculates KCDI, Shannon index and weighting factors.
- **Visualisations**: generates a KCDI compass chart and an intersectional bar chart.
- **Data processing**: cleans and preprocesses Scopus exports, infers gender, and classifies
  authors into Global North / Global South based on country information.
- **Data export**: allows downloading the processed dataset for further analysis.

### How to use

1. Upload a `.csv` or `.xlsx` file exported from Scopus.
2. The file must contain at least the columns `Authors` and `Country`.
3. The app will process the data and display the KCDI metrics and plots.

### Deployment

The app is deployed on **Streamlit Community Cloud**.  
You can also run it locally with:

```bash
pip install -r requirements.txt
streamlit run kcdi_app.py

### Dependencies

All required Python packages are listed in requirements.txt.


5. Write a commit message, e.g. `Translate README to English`.
6. Click **“Commit changes”**.

### 2. Verify the KRATOS app

For the new KRATOS app (`kratoskcdi.streamlit.app`):

- Repository must be `missmew33/kratos-streamlit-app`.
- Main file must be `app.py`.
- When you open it, the title should be, in English:  
  “KRATOS – Knowledge Justice Analytics for Scholarly Data”.

If that is what you see, then the KRATOS app is already fully in English; only the old KCDI repo README needed translation.
::contentReference[oaicite:0]{index=0}

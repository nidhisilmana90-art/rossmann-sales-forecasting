# Streamlit Deployment

## Local
```bash
cd deployment
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud
1. Push the project folder to GitHub.
2. Open Streamlit Community Cloud.
3. Select the GitHub repository and `deployment/app.py`.
4. Deploy.
5. Add the deployed URL to the final submission.

The dashboard accepts manual inputs or a CSV upload and provides predicted sales,
predicted customers, a chart, and CSV download.

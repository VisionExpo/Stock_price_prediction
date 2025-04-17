# Deploying the Stock Price Prediction App

This guide explains how to deploy the Stock Price Prediction application to Render.

## Prerequisites

- A [Render](https://render.com/) account
- Git repository with your project

## Deployment Steps

### Option 1: Deploy from the Render Dashboard

1. Log in to your Render account
2. Click on "New +" and select "Web Service"
3. Connect your Git repository
4. Configure the service:
   - Name: `stock-price-prediction` (or any name you prefer)
   - Environment: `Python`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
5. Select the appropriate plan (Free tier works for testing)
6. Click "Create Web Service"

### Option 2: Deploy using render.yaml

1. Make sure the `render.yaml` file is in your repository
2. Log in to your Render account
3. Go to the "Blueprints" section
4. Click on "New Blueprint Instance"
5. Connect your Git repository
6. Render will automatically detect the `render.yaml` file and configure the services
7. Click "Apply" to deploy

## Environment Variables

If you're using the Tiingo API, you'll need to set the `TIINGO_API_KEY` environment variable in your Render service:

1. Go to your web service in the Render dashboard
2. Click on "Environment"
3. Add the environment variable:
   - Key: `TIINGO_API_KEY`
   - Value: Your Tiingo API key
4. Click "Save Changes"

## Accessing Your Deployed App

Once deployed, you can access your app at the URL provided by Render, which will look something like:
`https://stock-price-prediction-xxxx.onrender.com`

## Troubleshooting

If you encounter any issues:

1. Check the logs in the Render dashboard
2. Make sure all dependencies are correctly listed in `requirements.txt`
3. Verify that the start command is correct
4. Ensure all necessary environment variables are set

## Local Testing

Before deploying, you can test the app locally:

```
streamlit run app.py
```

This will start the app on `http://localhost:8501`

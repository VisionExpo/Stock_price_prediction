# Deploying the Stock Price Prediction App

This guide explains how to deploy the Stock Price Prediction application to Render.

## Prerequisites

- A [Render](https://render.com/) account
- Git repository with your project
- Python 3.9+ environment

## System Requirements

The application requires certain system packages to be installed on the deployment server:
- graphviz (for model architecture visualization)
- build-essential (for compiling some Python packages)

These will be installed automatically through the build script.

## Deployment Steps

### Option 1: Deploy from the Render Dashboard

1. Log in to your Render account
2. Click on "New +" and select "Web Service"
3. Connect your Git repository
4. Configure the service:
   - Name: `stock-price-prediction` (or any name you prefer)
   - Environment: `Python 3.9`
   - Build Command:
     ```
     apt-get update -y && \
     apt-get install -y graphviz build-essential && \
     pip install -r requirements.txt
     ```
   - Start Command: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
   - Plan: Free (Starter) or Standard based on your needs
5. Environment Variables:
   - Add your Tiingo API key:
     - Key: `TIINGO_API_KEY`
     - Value: Your Tiingo API key
6. Click "Create Web Service"

### Option 2: Deploy using render.yaml

1. Ensure the `render.yaml` file is in your repository (it's already configured)
2. Log in to your Render account
3. Go to the "Blueprints" section
4. Click on "New Blueprint Instance"
5. Connect your Git repository
6. Render will automatically detect the `render.yaml` file and configure the services
7. Click "Apply" to deploy

## Resource Requirements

- Memory: Minimum 1GB RAM (2GB recommended)
- CPU: At least 1 vCPU
- Storage: Minimum 512MB free space

The free tier of Render should be sufficient for testing and light usage. For production use with multiple users, consider upgrading to the Standard plan.

## Performance Optimization

The application includes several optimizations for deployment:
1. Caching of stock data to reduce API calls
2. Batch prediction to minimize memory usage
3. Model compression for faster loading

## Monitoring and Maintenance

1. Monitor your application through the Render dashboard:
   - Check CPU and memory usage
   - Review application logs
   - Monitor response times

2. Regular maintenance tasks:
   - Update dependencies monthly
   - Review and clean cached data
   - Monitor model performance metrics

## Troubleshooting

If you encounter issues:

1. Check application logs in the Render dashboard
2. Common issues and solutions:
   - Memory errors: Increase the memory allocation
   - Timeout errors: Adjust the request timeout settings
   - Model loading issues: Clear the service cache
   - Stock data errors: Check Tiingo API status

## Local Testing

Before deploying, test the app locally:

```bash
# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

The app will be available at `http://localhost:8501`

## Health Check

The application automatically performs health checks on:
- Model availability
- Data API connectivity
- Memory usage
- Cache status

## Security Considerations

1. The application uses HTTPS by default on Render
2. Stock data is fetched through secure API calls
3. User inputs are sanitized to prevent injection attacks
4. No sensitive credentials are required

## Continuous Deployment with GitHub Actions

You can set up automatic deployments using GitHub Actions:

1. In your Render dashboard, go to your web service
2. Click on "Settings"
3. Scroll down to "Deploy Hooks"
4. Click "Create Deploy Hook" and copy the generated URL
5. In your GitHub repository:
   - Go to "Settings" > "Secrets and variables" > "Actions"
   - Click "New repository secret"
   - Name: `RENDER_DEPLOY_HOOK`
   - Value: Paste the deploy hook URL you copied from Render
   - Click "Add secret"

Now, whenever you push to the main branch or create a new tag, GitHub Actions will automatically trigger a deployment on Render.

## Scaling

To handle more users:
1. Upgrade to Render's Standard plan
2. Increase the memory allocation
3. Enable auto-scaling if needed
4. Consider implementing request queuing for heavy workloads

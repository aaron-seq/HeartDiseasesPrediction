# Heart Disease Prediction Tool

A comprehensive machine learning project for predicting heart disease risk using state-of-the-art neural network architecture. This repository includes both a Jupyter Notebook for model development and a production-ready FastAPI backend with React frontend.

## Features

- **Advanced ML Model**: Neural network with 97%+ accuracy on Cleveland Heart Disease dataset
- **Interactive UI**: Modern React frontend with real-time predictions
- **Production API**: FastAPI backend with comprehensive error handling and validation
- **Model Explainability**: SHAP values for understanding predictions
- **Containerized Deployment**: Docker and docker-compose for easy deployment

## Project Structure

```
.
├── backend/                  # FastAPI backend application
│   ├── app/
│   │   ├── api/             # API endpoints
│   │   ├── core/            # Configuration and logging
│   │   ├── models/          # ML models and schemas
│   │   ├── services/        # Business logic
│   │   └── utils/           # Utility functions
│   ├── tests/               # Test suite
│   ├── Dockerfile           # Backend container
│   └── requirements.txt     # Python dependencies
├── frontend/                 # React frontend application
├── Enhanced2025_Heart_Disease_Prediction.ipynb  # ML development notebook
└── docker-compose.yml        # Multi-container orchestration
```

## Prerequisites

### For Jupyter Notebook
- Python 3.11+
- Jupyter Notebook or JupyterLab

### For Backend API
- Python 3.11+
- Docker and Docker Compose (recommended)
- System dependencies: gcc, g++ (for some Python packages)

### For Frontend
- Node.js 18+ and npm

## Quick Start with Docker

The fastest way to run the entire application:

```bash
# Clone the repository
git clone https://github.com/aaron-seq/heart-diseases-prediction-tool.git
cd heart-diseases-prediction-tool

# Start all services
docker-compose up -d

# Access the application
# Frontend: http://localhost:5173
# Backend API: http://localhost:8000
# API Documentation: http://localhost:8000/docs
```

## Backend Setup

### Using Docker (Recommended)

```bash
cd backend

# Create .env file from template
cp .env.example .env
# Edit .env with your configuration

# Build and run
docker build -t heart-disease-api .
docker run -p 8000:8000 --env-file .env heart-disease-api
```

### Manual Setup

```bash
cd backend

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Configure MODEL_PATH and other variables in .env

# Run development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev

# Access at http://localhost:5173
```

## Jupyter Notebook Usage

For model development and experimentation:

```bash
# Install notebook dependencies
pip install jupyter notebook

# Start Jupyter
jupyter notebook Enhanced2025_Heart_Disease_Prediction.ipynb
```

The notebook includes:
- Data exploration and visualization
- Feature engineering
- Model training and evaluation
- SHAP explainability analysis
- Model export for production use

## API Documentation

Once the backend is running, access the interactive API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Example API Request

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 45,
    "sex": 1,
    "cp": 2,
    "trestbps": 120,
    "chol": 200,
    "fbs": 0,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.5,
    "slope": 1,
    "ca": 0,
    "thal": 2
  }'
```

## Data and Model Setup

### Option 1: Automatic Dataset Download

The application automatically fetches the Cleveland Heart Disease dataset from UCI ML Repository on first run.

### Option 2: Train from Scratch

1. Run the Jupyter notebook `Enhanced2025_Heart_Disease_Prediction.ipynb`
2. Model artifacts will be saved to the configured MODEL_PATH
3. Backend will load these artifacts on startup

## Running Tests

```bash
cd backend
pytest -v

# With coverage
pytest --cov=app --cov-report=html
```

## Deployment

### Using Render.com

The repository includes `render.yaml` for easy deployment:

1. Fork this repository
2. Connect your Render account
3. Create new Blueprint Instance
4. Select this repository

### Environment Variables

Required environment variables (see `.env.example`):

- `API_HOST`: API host address (default: 0.0.0.0)
- `API_PORT`: API port (default: 8000)
- `LOG_LEVEL`: Logging level (default: INFO)
- `ALLOWED_ORIGINS`: Comma-separated CORS origins
- `MODEL_PATH`: Path to model artifacts
- `SECRET_KEY`: Application secret key

## Model Performance

- **Accuracy**: 97%+
- **Dataset**: Cleveland Heart Disease (UCI ML Repository)
- **Features**: 13 clinical features
- **Algorithm**: Enhanced Neural Network with regularization

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -am 'Add your feature'`
4. Push to branch: `git push origin feature/your-feature`
5. Submit a Pull Request

Please ensure:
- Code follows project style guidelines
- Tests pass: `pytest`
- Documentation is updated
- Commit messages are clear and descriptive

## Troubleshooting

### Model Not Found Error
- Ensure MODEL_PATH is correctly set in .env
- Run the Jupyter notebook to train and save the model
- Verify model files exist in the specified path

### Docker Build Issues
- Clear Docker cache: `docker system prune -a`
- Rebuild without cache: `docker-compose build --no-cache`

### Import Errors
- Verify all __init__.py files are present
- Check Python version: `python --version` (should be 3.11+)
- Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`

## License

MIT License - see LICENSE file for details

## Acknowledgments

- UCI Machine Learning Repository for the Cleveland Heart Disease dataset
- FastAPI and Pydantic teams for excellent frameworks
- TensorFlow and scikit-learn communities

## Contact

Aaron Sequeira - [GitHub](https://github.com/aaron-seq)

## Citation

If you use this project in your research, please cite:

```
@software{sequeira2025heart,
  author = {Sequeira, Aaron},
  title = {Heart Disease Prediction Tool},
  year = {2025},
  url = {https://github.com/aaron-seq/heart-diseases-prediction-tool}
}
```

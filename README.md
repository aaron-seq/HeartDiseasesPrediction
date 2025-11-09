# Heart Disease Prediction using Advanced Neural Networks

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg) ![Keras](https://img.shields.io/badge/Keras-2.x-red.svg) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-green.svg) ![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688.svg) ![React](https://img.shields.io/badge/React-18-61DAFB.svg) ![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A production-ready machine learning system for predicting cardiovascular disease risk using deep neural networks. This full-stack application combines advanced ML techniques with a modern web interface to deliver real-time heart disease risk assessments.

## Technical Stack

### Backend
- **Framework**: FastAPI with async/await support for high-performance API endpoints
- **ML Pipeline**: TensorFlow/Keras for neural network inference
- **Data Processing**: NumPy, Pandas for efficient data manipulation
- **Model Serving**: Custom ML service with batch prediction capabilities
- **Logging**: Structured logging with Loguru
- **Validation**: Pydantic models for request/response validation

### Frontend  
- **Framework**: React 18 with TypeScript for type-safe development
- **Build Tool**: Vite for fast development and optimized production builds
- **Styling**: Tailwind CSS for utility-first responsive design
- **State Management**: React Hooks for component state
- **API Client**: Axios with interceptors for centralized error handling

### Infrastructure
- **Containerization**: Docker and Docker Compose for consistent environments
- **API Documentation**: Interactive OpenAPI/Swagger UI
- **Testing**: Pytest for backend, React Testing Library for frontend
- **CI/CD**: GitHub Actions for automated testing and deployment

## Architecture Overview

The system follows a microservices architecture with clear separation of concerns:

```
frontend/          # React TypeScript application
├── src/
│   ├── components/    # Reusable UI components
│   ├── services/      # API client services
│   ├── types/         # TypeScript type definitions
│   └── App.tsx        # Main application component
│
backend/           # FastAPI application
├── app/
│   ├── api/endpoints/ # REST API route handlers  
│   ├── core/          # Configuration and logging
│   ├── models/        # Pydantic data models
│   ├── services/      # Business logic and ML inference
│   └── utils/         # Utility functions
│
notebooks/         # Jupyter notebooks for ML experiments
```

## Key Features

### Advanced Machine Learning
- Deep neural network with batch normalization and dropout regularization
- Feature engineering pipeline with scaling and imputation
- Model achieves approximately 87% accuracy and 0.92 ROC-AUC score
- SHAP integration for explainable AI and model interpretability

### Production-Ready API
- RESTful endpoints with proper HTTP status codes and error handling
- Request validation using Pydantic models
- CORS middleware for secure cross-origin requests
- Health check endpoints for monitoring
- Async request processing for improved throughput

### Modern Web Interface
- Responsive design that works on desktop and mobile devices
- Real-time prediction results with confidence scores
- Interactive form with client-side validation
- Loading states and error handling for better UX
- TypeScript for compile-time type checking

### Developer Experience
- Hot module replacement in development mode
- Comprehensive type definitions for IDE autocomplete
- Structured logging for debugging
- Docker Compose for one-command development environment
- Environment-based configuration

## Performance Metrics

| Metric | Score |
|--------|-------|
| Accuracy | ~87% |
| ROC-AUC | ~0.92 |
| Precision | ~0.85 |
| Recall | ~0.88 |
| API Response Time | <100ms |
| Frontend Bundle Size | ~150KB (gzipped) |

## Prerequisites

- Python 3.8 or higher
- Node.js 16 or higher  
- Docker and Docker Compose (optional, for containerized deployment)
- 4GB RAM minimum for ML model inference

## Quick Start

### Using Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/aaron-seq/HeartDiseasesPrediction.git
cd HeartDiseasesPrediction

# Start all services
docker-compose up -d

# Access the application
# Frontend: http://localhost:5173
# Backend API: http://localhost:8000
# API Documentation: http://localhost:8000/docs
```

### Manual Setup

#### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env

# Run development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

## API Endpoints

### Health Check
```http
GET /api/v1/health
```
Returns service health status and version information.

### Prediction
```http
POST /api/v1/predict
Content-Type: application/json

{
  "age": 55,
  "sex": 1,
  "chest_pain_type": 2,
  "resting_blood_pressure": 130,
  "cholesterol": 250,
  "fasting_blood_sugar": 0,
  "resting_ecg": 1,
  "max_heart_rate": 150,
  "exercise_induced_angina": 0,
  "st_depression": 1.5,
  "st_slope": 2,
  "num_major_vessels": 1,
  "thalassemia": 2
}
```
Returns prediction result with probability scores.

## Deployment

### Vercel (Frontend)

```bash
cd frontend
npm run build
vercel deploy --prod
```

### Render (Backend)

1. Connect your GitHub repository to Render
2. Select "Web Service" as the service type
3. Set build command: `pip install -r backend/requirements.txt`
4. Set start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Deploy

### Railway

Both frontend and backend can be deployed using Railway's GitHub integration with the provided configuration files.

## Testing

### Backend Tests

```bash
cd backend
pytest tests/ -v --cov=app --cov-report=html
```

### Frontend Tests

```bash
cd frontend  
npm run test
npm run test:coverage
```

## Environment Variables

### Backend (.env)

```env
API_TITLE=Heart Disease Prediction API
API_VERSION=1.0.0
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
ALLOWED_ORIGINS=http://localhost:5173,http://localhost:3000
MODEL_PATH=./models/heart_disease_model.h5
```

### Frontend (.env)

```env
VITE_API_BASE_URL=http://localhost:8000
VITE_API_TIMEOUT=10000
```

## Model Details

### Architecture

- Input Layer: 13 features (patient health metrics)
- Hidden Layers: 3 dense layers with batch normalization
  - Layer 1: 128 neurons, ReLU activation, 30% dropout  
  - Layer 2: 64 neurons, ReLU activation, 20% dropout
  - Layer 3: 32 neurons, ReLU activation, 20% dropout
- Output Layer: 1 neuron, Sigmoid activation for binary classification

### Training Details

- Dataset: UCI Cleveland Heart Disease dataset (303 samples)
- Train/Test Split: 80/20 with stratification
- Optimization: Adam optimizer with learning rate scheduling
- Loss Function: Binary crossentropy
- Early Stopping: Monitoring validation loss with patience of 15 epochs
- Data Augmentation: SMOTE for handling class imbalance

### Feature Engineering

- StandardScaler for continuous variables
- SimpleImputer for handling missing values
- Feature correlation analysis to identify redundant features
- Dimensionality reduction considerations

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with clear commit messages
4. Add tests for new functionality
5. Ensure all tests pass (`pytest` for backend, `npm test` for frontend)
6. Update documentation as needed
7. Submit a pull request with a detailed description

### Code Style

- Backend: Follow PEP 8 guidelines, use Black for formatting
- Frontend: Use ESLint and Prettier for consistent code style
- Commit messages: Follow conventional commits specification

## Monitoring and Logging

- Structured JSON logging for easy parsing and analysis
- Request/response logging with correlation IDs
- Error tracking with detailed stack traces
- Performance metrics collection
- Health check endpoints for uptime monitoring

## Security Considerations

- Input validation on both client and server
- CORS configuration for allowed origins
- Rate limiting on API endpoints (recommended for production)
- Environment-based configuration for sensitive data
- HTTPS enforcement in production
- Regular dependency updates for security patches

## Roadmap

- [ ] Add user authentication and authorization
- [ ] Implement prediction history tracking
- [ ] Add batch prediction support via CSV upload
- [ ] Integrate additional ML models for ensemble predictions
- [ ] Add real-time monitoring dashboard
- [ ] Implement A/B testing framework
- [ ] Add internationalization support
- [ ] Create mobile app versions (React Native)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- UCI Machine Learning Repository for the Cleveland Heart Disease dataset
- TensorFlow and Keras teams for the ML framework
- FastAPI community for the excellent web framework
- React and Vite teams for the frontend tools

## Citation

If you use this project in your research, please cite:

```bibtex
@software{heart_disease_prediction,
  author = {Aaron Sequeira},
  title = {Heart Disease Prediction using Advanced Neural Networks},
  year = {2024},
  url = {https://github.com/aaron-seq/HeartDiseasesPrediction}
}
```

## Support

For questions, issues, or feature requests, please open an issue on GitHub or contact the maintainers.

---

Built with care for advancing cardiovascular health prediction through machine learning.

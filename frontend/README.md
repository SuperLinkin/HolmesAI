# Holmes AI Dashboard

Modern React dashboard for monitoring and managing the Holmes AI transaction categorization engine.

## Features

- **Real-time Dashboard**: Overview of system health, predictions, and performance metrics
- **Performance Monitoring**: Live drift detection and model performance tracking
- **Transaction Categorization**: Interactive interface for categorizing transactions
- **Feedback Management**: Submit corrections for continuous learning
- **Analytics**: Comprehensive insights into model performance and category distribution

## Tech Stack

- React 18
- Vite
- React Router
- React Query
- Recharts (for visualizations)
- Tailwind CSS
- Lucide React (icons)
- Axios

## Prerequisites

- Node.js 16+
- npm or yarn
- Holmes AI backend running on port 8000

## Installation

```bash
cd frontend
npm install
```

## Development

```bash
# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

The dashboard will be available at `http://localhost:3000`

## Environment Variables

Create a `.env` file in the frontend directory:

```bash
VITE_API_URL=http://localhost:8000
```

## Features Overview

### Dashboard
- System health status
- Real-time prediction statistics
- Category distribution
- Model information
- Recent activity

### Monitoring
- Drift detection alerts
- Performance metrics (F1, Precision, Recall)
- Historical trends
- Threshold monitoring

### Categorize
- Interactive transaction input form
- Real-time categorization
- Confidence scores
- Probability breakdown

### Feedback
- Submit corrections
- View correction statistics
- Track retraining progress
- Most corrected categories

### Analytics
- Category distribution charts
- Feature importance analysis
- Model statistics
- System performance metrics

## API Integration

The dashboard communicates with the Holmes AI backend through a REST API. All API calls are proxied through Vite's dev server to avoid CORS issues during development.

## Production Deployment

1. Build the frontend:
```bash
npm run build
```

2. Serve the `dist` folder with your preferred web server (nginx, Apache, etc.)

3. Update `VITE_API_URL` to point to your production API

## License

Copyright © 2025 Holmes AI Team

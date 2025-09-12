# Docker Setup Guide

## Local Testing

To test the Docker build locally:

```bash
# Build the image
docker build -t solar-wind-app:latest .

# Run the container
docker run -p 8501:8501 solar-wind-app:latest

# Access the app at http://localhost:8501
```

## GitHub Actions Setup

### Required Secrets

Add these secrets to your GitHub repository (Settings → Secrets and variables → Actions):

1. **DOCKER_USERNAME**: Your DockerHub username
2. **DOCKER_PASSWORD**: Your DockerHub access token (recommended) or password

### Creating a DockerHub Access Token

1. Log in to [DockerHub](https://hub.docker.com)
2. Go to Account Settings → Security
3. Click "New Access Token"
4. Give it a descriptive name (e.g., "GitHub Actions")
5. Copy the token and add it as `DOCKER_PASSWORD` secret

### Repository Setup

1. Push your code to GitHub
2. The workflow will automatically:
   - Build on every push to main/master
   - Create multi-architecture images (amd64, arm64)
   - Push to DockerHub with proper tags
   - Generate build attestations for security

### Docker Image Tags

The workflow creates these tags:
- `latest` (from main/master branch)
- `<branch-name>` (for branch pushes)  
- `v1.0.0` (for version tags like `v1.0.0`)
- `1.0` (major.minor for version tags)

### Usage

Once deployed, your image will be available at:
```
docker.io/<your-username>/solar-wind-app:latest
```

Run it with:
```bash
docker run -p 8501:8501 <your-username>/solar-wind-app:latest
```
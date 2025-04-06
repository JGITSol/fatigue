# TODO List

## Containerization

- [ ] Set up Docker container for the application
  - [ ] Create a `Dockerfile` with Python 3.11 as the base image
  - [ ] Include all dependencies from requirements.txt
  - [ ] Configure appropriate environment variables
  - [ ] Set up proper entrypoint for the application

- [ ] Implement Docker Compose setup
  - [ ] Create `docker-compose.yml` file
  - [ ] Define services (app, potentially database if needed)
  - [ ] Configure networking between services
  - [ ] Set up volume mounts for persistent data
  - [ ] Add health checks for services

## Model Management

- [ ] Document model sources from Huggingface
  - [ ] Pin specific model versions/commits in documentation
  - [ ] Add links to original Huggingface repositories:
    - Example: https://huggingface.co/models?pipeline_tag=face-detection
  - [ ] Document model licenses and attribution requirements
  - [ ] Create script for reproducible model downloads

## Testing

- [ ] Implement real data testing on video materials
  - [ ] Create a test dataset of diverse video samples
  - [ ] Develop benchmark tests for detection accuracy
  - [ ] Measure and document performance metrics
  - [ ] Test edge cases (low light, different angles, etc.)

- [ ] Set up comprehensive test suite
  - [ ] Unit tests for core functionality
  - [ ] Integration tests for the complete pipeline
  - [ ] Performance tests for resource usage
  - [ ] Regression tests for previously fixed issues

- [ ] Implement test coverage reporting
  - [ ] Configure pytest-cov for coverage reports
  - [ ] Set minimum coverage thresholds (aim for >80%)
  - [ ] Integrate coverage reporting with CI/CD pipeline

## Documentation

- [ ] Enhance existing documentation
  - [ ] Add detailed API documentation
  - [ ] Include deployment instructions for different environments
  - [ ] Create troubleshooting guide
  - [ ] Document configuration options

- [ ] Add developer documentation
  - [ ] Code style guidelines
  - [ ] Contribution workflow
  - [ ] Architecture diagrams
  - [ ] Development environment setup instructions
# GuPT - Gothenburg University RAG System
# Multi-stage Docker build for production-ready container
# Uses environment.yml for dependencies

FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy environment file and install dependencies
COPY environment.yml .
# Extract pip dependencies from environment.yml and install them
RUN grep -A 100 "pip:" environment.yml | grep "    -" | sed 's/    - //' > requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash gupt
RUN chown -R gupt:gupt /app
USER gupt

# Expose port
EXPOSE 7860

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:7860/ || exit 1

# Default command
CMD ["python", "src/main.py"] 
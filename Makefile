# GuPT Makefile
# Cross-platform setup for GuPT RAG System

.PHONY: help install setup clean test run docker-build docker-run docker-stop docker-clean docker-logs docker-restart docker-shell

# Default target
help:
	@echo "GuPT - Gothenburg University RAG System"
	@echo "========================================"
	@echo ""
	@echo "🐳 Docker Commands (Recommended - Simple & Universal):"
	@echo "  make docker-build   - Build Docker image"
	@echo "  make docker-run     - Run with Docker Compose"
	@echo "  make docker-stop    - Stop Docker containers"
	@echo "  make docker-clean   - Remove Docker containers and images"
	@echo "  make docker-logs    - View application logs"
	@echo ""
	@echo "📦 Conda Commands (Development Environment):"
	@echo "  make install        - Create conda environment and install dependencies"
	@echo "  make setup          - Complete setup (environment + .env file)"
	@echo "  make clean          - Remove conda environment"
	@echo "  make test           - Test the installation"
	@echo "  make run            - Run the application"
	@echo ""

# Install dependencies
install:
	@echo "📦 Creating conda environment..."
	conda env create -f environment.yml
	@echo "✅ Environment created! Activate with: conda activate gupt"

# Complete setup
setup: install
	@echo "📝 Setting up environment file..."
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "✅ .env file created from .env.example"; \
		echo "⚠️  Please edit .env and add your OpenAI API key"; \
	else \
		echo "⚠️  .env file already exists"; \
	fi
	@echo ""
	@echo "🎯 Setup complete! Next steps:"
	@echo "1. conda activate gupt"
	@echo "2. Edit .env file with your API keys"
	@echo "3. make run"

# Clean up
clean:
	@echo "🧹 Removing conda environment..."
	conda env remove -n gupt
	@echo "✅ Environment removed"

# Test installation
test:
	@echo "🧪 Testing installation..."
	@echo "⚠️  Make sure to run: conda activate gupt"
	@echo "Then manually test with: python -c \"import langchain, openai, gradio, chromadb, posthog; print('✅ All packages imported successfully')\""
	@echo "✅ Test instructions provided!"

# Run the application
run:
	@echo "🚀 Starting GuPT..."
	@echo "⚠️  Make sure to run: conda activate gupt"
	@echo "Then run: python src/main.py"

# Docker commands
docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t gupt:latest .
	@echo "✅ Docker image built successfully!"

docker-run:
	@echo "🚀 Starting GuPT with Docker Compose..."
	@if [ ! -f .env ]; then \
		echo "⚠️  .env file not found. Creating from template..."; \
		cp .env.example .env; \
		echo "🔧 Please edit .env file with your OpenAI API key"; \
		echo "Then run 'make docker-run' again"; \
		exit 1; \
	fi
	docker compose up -d
	@echo "✅ GuPT is running at http://localhost:7860"

docker-stop:
	@echo "🛑 Stopping Docker containers..."
	docker compose down

docker-clean:
	@echo "🧹 Cleaning up Docker containers and images..."
	docker compose down --rmi all --volumes --remove-orphans
	@echo "🗑️  Removing GuPT images..."
	-docker rmi gupt:latest gupt-app:latest 2>/dev/null || true
	@echo "🧹 Cleaning up unused Docker resources..."
	docker system prune -f
	@echo "✅ Docker cleanup completed!"

docker-logs:
	@echo "📋 Viewing application logs..."
	docker compose logs -f gupt

docker-restart:
	@echo "🔄 Restarting Docker containers..."
	docker compose restart

docker-shell:
	@echo "🐚 Opening shell in Docker container..."
	docker compose exec gupt bash 
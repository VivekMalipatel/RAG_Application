.PHONY: help build up down logs clean dev dev-down dev-logs

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

build: ## Build the Docker containers
	docker-compose build

up: ## Start the production services
	docker-compose up -d

down: ## Stop the production services
	docker-compose down

logs: ## View production logs
	docker-compose logs -f

clean: ## Clean up containers, networks, and volumes
	docker-compose down -v
	docker system prune -f

dev: ## Start development services with hot reload
	docker-compose -f docker-compose.dev.yml up -d

dev-down: ## Stop development services
	docker-compose -f docker-compose.dev.yml down

dev-logs: ## View development logs
	docker-compose -f docker-compose.dev.yml logs -f

status: ## Show container status
	docker-compose ps

restart: ## Restart production services
	docker-compose restart

shell: ## Access the main container shell
	docker exec -it agent-api bash

health: ## Check API health
	curl -f http://localhost:8000/health || echo "API not ready"

setup: ## Initial setup - copy env file
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env file from example. Please update it with your configuration."; \
	else \
		echo ".env file already exists"; \
	fi

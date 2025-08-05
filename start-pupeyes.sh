#!/bin/bash

# PupEyes Docker Startup Script
# This script helps users easily start and manage the PupEyes Docker container

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
}

# Check if docker-compose is installed
check_docker_compose() {
    if ! command -v docker-compose &> /dev/null; then
        print_error "docker-compose is not installed. Please install docker-compose first."
        exit 1
    fi
}

# Create data directory if it doesn't exist
create_data_dir() {
    if [ ! -d "./data" ]; then
        print_status "Creating data directory..."
        mkdir -p ./data
        print_success "Data directory created at ./data"
    fi
}

# Build and start the container
start_container() {
    print_status "Building and starting PupEyes container..."
    docker-compose up --build -d
    
    if [ $? -eq 0 ]; then
        print_success "PupEyes container started successfully!"
        print_status "Jupyter Lab is available at: http://localhost:8888"
        print_status "Press Ctrl+C to stop the container"
        
        # Show logs
        docker-compose logs -f
    else
        print_error "Failed to start container"
        exit 1
    fi
}

# Stop the container
stop_container() {
    print_status "Stopping PupEyes container..."
    docker-compose down
    print_success "Container stopped"
}

# Show container status
status() {
    print_status "Container status:"
    docker-compose ps
}

# Show logs
logs() {
    print_status "Showing container logs:"
    docker-compose logs -f
}

# Clean up
cleanup() {
    print_warning "This will remove the container and all data. Are you sure? (y/N)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        print_status "Cleaning up..."
        docker-compose down -v --rmi all
        print_success "Cleanup completed"
    else
        print_status "Cleanup cancelled"
    fi
}

# Main script logic
case "${1:-start}" in
    "start")
        check_docker
        check_docker_compose
        create_data_dir
        start_container
        ;;
    "stop")
        stop_container
        ;;
    "restart")
        stop_container
        sleep 2
        start_container
        ;;
    "status")
        status
        ;;
    "logs")
        logs
        ;;
    "cleanup")
        cleanup
        ;;
    "help"|"-h"|"--help")
        echo "PupEyes Docker Management Script"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  start     Build and start the container (default)"
        echo "  stop      Stop the container"
        echo "  restart   Restart the container"
        echo "  status    Show container status"
        echo "  logs      Show container logs"
        echo "  cleanup   Remove container and all data"
        echo "  help      Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0 start    # Start the container"
        echo "  $0 stop     # Stop the container"
        echo "  $0 logs     # View logs"
        ;;
    *)
        print_error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac 
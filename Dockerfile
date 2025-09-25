# Use official Python base image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies (optional: for pandas performance)
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files into container
COPY backend/ /app

EXPOSE 8000

# Default command to run your script (change main.py to your script file)
# CMD ["python", "main.py"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

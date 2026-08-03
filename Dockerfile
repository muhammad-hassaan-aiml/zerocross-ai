# Use a lightweight Python 3.12 image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for C++ compilation
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the repository into the container
COPY . .

# Grant execution permissions and build the C++ engine
RUN chmod +x build_kaggle.sh
RUN ./build_kaggle.sh

# Expose the default port used by Hugging Face Spaces
EXPOSE 7860

# Command to start the FastAPI server
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "7860"]
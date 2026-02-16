# Use an official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.9-slim

# Set environment variables to prevent Python from writing pyc files to disc
# and buffering stdout and stderr.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container.
WORKDIR /app

# Copy the requirements file into the container.
COPY requirements.txt .

# Install dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code.
COPY . .

# Expose the port that Uvicorn will run on.
EXPOSE 8000

# Command to run the application using Uvicorn.
# Using 0.0.0.0 for host to be accessible externally within container networks (like Render/Railway).
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

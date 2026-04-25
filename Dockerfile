FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install streamlit

# Copy the rest of the application
COPY . .

# Default command runs the trader
CMD ["python", "main.py", "--mode", "trade", "--strategy", "adaptive"]

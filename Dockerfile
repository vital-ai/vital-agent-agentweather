FROM python:3.11-slim

WORKDIR /usr/src/app

RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /usr/src/app/

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /var/log/agentcontainer

RUN chmod -R 755 /var/log/agentcontainer

# Make port 7007 available to the world outside this container
EXPOSE 7007

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7007"]


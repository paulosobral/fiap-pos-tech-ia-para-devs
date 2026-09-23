FROM public.ecr.aws/docker/library/python:3.11-slim

WORKDIR /app

# Dependências do sistema (compilação leve se necessário)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV PORT=8080

EXPOSE 8080

CMD ["python", "server.py"]

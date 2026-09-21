FROM public.ecr.aws/docker/library/python:3.11-slim

WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py app.py

EXPOSE 80
CMD ["streamlit", "run", "app.py", "--server.port", "80", "--server.address", "0.0.0.0", "--server.headless", "true"]
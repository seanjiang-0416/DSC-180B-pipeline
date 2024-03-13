FROM python:3.9-slim
EXPOSE 8080
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt


COPY . .

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
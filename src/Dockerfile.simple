FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY src /app/src
CMD ["uvicorn","src.app:app","--host","0.0.0.0","--port","8080","--workers","2"]

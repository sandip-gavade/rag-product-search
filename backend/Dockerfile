FROM python:3.12-slim

WORKDIR /app

# psycopg[binary] and pgvector's Python client are both pure wheels — no
# compiler toolchain needed, keeps the image small.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]

from python:3.7-slim

WORKDIR /app

COPY . .

RUN pip install -- upgrade pip \
    && pip install --trusted-host pypi.python.org --requirement requirements.txt

CMD ["python", "main.py"]

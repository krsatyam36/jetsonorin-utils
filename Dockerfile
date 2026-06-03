FROM nvcr.io/nvidia/l4t-pytorch:r36.3.0p0

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt pyproject.toml ./
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python3", "src/stream.py"]

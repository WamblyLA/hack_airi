FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime
WORKDIR /submission
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY README.md .
ENV PYTHONPATH=/submission
CMD ["python", "-m", "src.train", "--help"]

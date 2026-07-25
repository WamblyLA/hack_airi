FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime
WORKDIR /submission
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY checkpoints ./checkpoints
COPY bitstreams ./bitstreams
COPY metrics ./metrics
COPY plots ./plots
COPY examples ./examples
COPY manifests ./manifests
COPY README.md .
ENV PYTHONPATH=/submission
CMD ["python", "-m", "src.decode", "--checkpoint", "checkpoints/checkpoint.pt", "--input", "bitstreams/sample_base_64x.bin", "--static", "examples/sample_input.npz", "--output", "examples/reconstruction.npy"]

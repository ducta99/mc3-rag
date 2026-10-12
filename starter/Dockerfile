# ==============================================================================
# AMD AI League — Mini-Challenge 3 (MC-3) Submission Dockerfile
#
# Constraints & Hard Gates:
# 1. Base image: Mandated rocm/pytorch build (layer identity verified).
# 2. Image size: <= 60 GiB uncompressed (base is ~29.2 GiB).
# 3. VRAM: 1 - 48 GiB continuously on GPU (MI300X gfx942 or Navi 31 gfx1100).
# 4. Outbound network: NONE at runtime (--network none).
# 5. Timing: <= 10 min startup/index, <= 30 sec per question.
# ==============================================================================

FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

WORKDIR /app

# Ensure non-interactive environment
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV MC3_INDEX_DIR=/app/index
ENV MC2_OUTPUT_DIR=/app/output
ENV MC3_MODELS_DIR=/models

# Install application dependencies without replacing ROCm torch
COPY app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Verify that pip did not overwrite ROCm PyTorch with CUDA PyTorch
RUN python3 -c "import torch; print('PyTorch build:', torch.__version__); assert '+rocm' in torch.__version__, 'ERROR: PyTorch ROCm was overwritten!'"

# Copy application source code
COPY app/ /app/

# Create required directories
RUN mkdir -p /app/corpus /app/output /app/index

# Start the GPU resident daemon upon container boot.
# This pre-allocates model parameters in GPU VRAM (1-48 GiB band) and
# serves queries through /tmp/mc3_daemon.sock in under 1 second.
CMD ["python3", "/app/server.py"]

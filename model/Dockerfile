FROM pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=1000

WORKDIR /app

COPY model/requirements.txt model/requirements.txt
RUN grep -vE '^(torch|torchvision)' model/requirements.txt > /tmp/model-requirements.txt \
    && pip install --no-cache-dir --retries 10 --break-system-packages -r /tmp/model-requirements.txt

COPY model model
COPY shared shared

EXPOSE 8001

CMD ["python", "-m", "uvicorn", "model.src.service:app", "--host", "0.0.0.0", "--port", "8001"]

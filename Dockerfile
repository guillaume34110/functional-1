FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir numpy==2.2.6 Pillow==12.0.0
COPY play.py run.py model.py token_model.py scene_model.py counter_model.py collect.py native_render.py functional_two.py context.json ./
COPY web/index.html web/index.html
COPY releases/functional-1/ releases/functional-1/
COPY releases/functional-2/ releases/functional-2/
RUN cd releases/functional-1 && sha256sum -c SHA256SUMS
RUN cd releases/functional-2 && sha256sum -c SHA256SUMS
USER 10001
EXPOSE 8765
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/health', timeout=3)"
CMD ["python", "play.py", "--host", "0.0.0.0", "--functional-2", "releases/functional-2", "--compare"]

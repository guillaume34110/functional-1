FROM node:22-alpine AS browser-runtime
WORKDIR /runtime
RUN npm pack --silent onnxruntime-web@1.31.1 && \
    tar -xzf onnxruntime-web-1.31.1.tgz \
      package/dist/ort.wasm.min.mjs \
      package/dist/ort-wasm-simd-threaded.mjs \
      package/dist/ort-wasm-simd-threaded.wasm

FROM python:3.12-slim AS export
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir numpy==2.2.6 Pillow==12.0.0 onnx==1.22.0
COPY browser_assets.py browser_export.py play.py run.py model.py token_model.py scene_model.py counter_model.py collect.py native_render.py functional_two.py context.json ./
COPY web/ web/
COPY releases/functional-1/ releases/functional-1/
COPY releases/functional-2/ releases/functional-2/
COPY releases/functional-3/ releases/functional-3/
RUN (cd releases/functional-1 && sha256sum -c SHA256SUMS) && \
    (cd releases/functional-2 && sha256sum -c SHA256SUMS) && \
    (cd releases/functional-3 && sha256sum -c SHA256SUMS) && \
    PYTHONWARNINGS=ignore python browser_export.py /site && \
    gzip -k /site/models/*.onnx /site/models/v3-color.json

FROM nginxinc/nginx-unprivileged:1.28-alpine
COPY web/nginx.conf /etc/nginx/nginx.conf
COPY --from=export /site/ /usr/share/nginx/html/
COPY --from=browser-runtime /runtime/package/dist/ /usr/share/nginx/html/vendor/
EXPOSE 8765
HEALTHCHECK CMD wget -q -O /dev/null http://127.0.0.1:8765/health

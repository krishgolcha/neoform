FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /repo

COPY . .
RUN pip install --no-cache-dir . ./extensions/neoform

WORKDIR /repo/extensions/neoform
EXPOSE 8787
CMD ["neoform", "serve", "--host", "0.0.0.0", "--port", "8787"]

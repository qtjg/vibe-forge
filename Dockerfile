# vibe-forge — dashboard + router in one container

FROM python:3.11-slim

# Docker builds with no access to a local model dir; install from source.
COPY . /app
WORKDIR /app

RUN pip install --no-cache-dir .

# Serve the dashboard; routing happens against the host's Ollama,
# so run with: docker run --network host -p 8420:8420 vibeforge
EXPOSE 8420

CMD ["vibeforge", "serve"]
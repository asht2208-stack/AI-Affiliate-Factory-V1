# --- AI Automation Affiliate V3: production container image ---
#
# Builds a lean, production-ready image for the FastAPI backend.
# Works with any container host (Render, Railway, Fly.io, etc.) that
# assigns a port via the $PORT environment variable.

FROM python:3.12-slim

# Prevents Python from writing .pyc files and buffering stdout/stderr,
# which matters for seeing logs promptly on a hosting platform's
# dashboard rather than them sitting in a buffer.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install OS-level dependencies Pillow needs to handle the full range
# of image formats merchant feeds actually send (JPEG, WebP, etc.).
# Installed before copying application code so this layer is cached
# and only re-run when the OS dependency list itself changes, not on
# every code change.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo \
    zlib1g \
    libwebp7 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies as a separate layer from the
# application code, so Docker's build cache can skip reinstalling
# every dependency when only application code changed, not
# requirements.txt.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the actual application code.
COPY app/ ./app/
COPY migrations/ ./migrations/
COPY alembic.ini .

# Most hosts (Render, Railway, Fly.io) inject the actual port to
# listen on via the $PORT environment variable at runtime rather than
# a fixed value -- this default is only used for local `docker run`
# testing where nothing else sets $PORT.
ENV PORT=8000
EXPOSE 8000

# Run any pending database migrations, then start the server. Using
# shell form (not exec form) specifically so $PORT is expanded by the
# shell at container start -- exec form would pass the literal string
# "$PORT" to uvicorn instead of its actual value.
CMD alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port $PORT


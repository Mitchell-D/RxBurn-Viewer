# syntax=docker/dockerfile:1
FROM python:3.11-slim

## Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gdal-bin \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

## Set dir
WORKDIR /app

## Install python dependencies
RUN pip install --no-cache-dir \
    zarr \
    uvicorn[standard] \
    fastapi[standard] \
    rasterio \
    starlette-compress \
    redis[asyncio]

## Expose the app on port 81
EXPOSE 81

## Start FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "81", "--workers", "8", "--reload"]

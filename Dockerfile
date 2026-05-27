FROM python:3.10-slim

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    GDAL_CONFIG=/usr/bin/gdal-config \
    CPLUS_INCLUDE_PATH=/usr/include/gdal \
    C_INCLUDE_PATH=/usr/include/gdal

WORKDIR /app

COPY requirements.txt ./requirements.txt
COPY pysteps_destine ./pysteps_destine

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cdo \
    libeccodes0 \
    libgomp1 \
    gdal-bin \
    libgdal-dev \
    && pip install --no-cache-dir --timeout 300 -r requirements.txt \
    && apt-get purge -y build-essential \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* /root/.cache /tmp/*

COPY nowcast_blend ./nowcast_blend
COPY configs ./configs
COPY resources ./resources

CMD ["python", "-m", "nowcast_blend.main"]

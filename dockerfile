FROM python:3.11-slim

WORKDIR /app
COPY src/ src/
COPY data/models/zone_hour_means_29.csv data/models/
COPY Makefile .

RUN apt-get update \
 && apt-get install -y --no-install-recommends make \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir pandas numpy

CMD ["/bin/bash"]
FROM --platform=linux/amd64 docker.io/library/python:3.11-slim

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED 1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
RUN apt-get update && apt-get install -y nano g++
USER user

WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app/
# COPY --chown=user:user ground_truth /opt/app/ground_truth

# You can add any Python dependencies to requirements.txt
RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

COPY --chown=user:user .totalsegmentator /home/user/.totalsegmentator
COPY --chown=user:user TotalSegmentator /opt/app/
COPY --chown=user:user nnUNet /opt/app/
COPY --chown=user:user evaluate.py /opt/app/
COPY --chown=user:user image_metrics.py /opt/app/
COPY --chown=user:user segmentation_metrics.py /opt/app/
COPY --chown=user:user ts_utils.py /opt/app/

ENTRYPOINT ["python", "evaluate.py"]

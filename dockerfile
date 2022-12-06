FROM tensorflow/tensorflow:latest-gpu

ENV TZ=Asia/Singapore
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install libsndfile1 (linux soundfile package)
RUN apt-get update && apt-get install -y gcc libsndfile1 ffmpeg wget \
    && rm -rf /var/lib/apt/lists/*

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

RUN python -m pip install --upgrade --no-cache-dir pip
RUN python -m pip install --no-cache-dir matplotlib soundfile scipy pyyaml pydub librosa auditok tqdm

WORKDIR /dent
RUN ["bash"]
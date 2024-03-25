FROM python:3.11

RUN apt-get update && apt-get install -y \
    python3-pyaudio \
    portaudio19-dev \
    liblapack-dev \
    libblas-dev \
    libsuitesparse-dev \
    libglpk-dev \
    libfftw3-dev \
    libsndfile1 \
    ffmpeg

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

# fixes cvxopt build
ENV CPPFLAGS="-I/usr/include/suitesparse"

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app

CMD ["uvicorn", "app.main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "80"]
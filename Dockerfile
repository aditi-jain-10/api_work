FROM python:3.10

ENV VENV_PATH="/venv"
ENV PATH="$VENV_PATH/bin:$PATH"

WORKDIR /app

RUN apt-get update && apt-get upgrade && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --upgrade poetry
RUN python -m venv /venv


COPY . .

RUN poetry build && \
    /venv/bin/pip install --upgrade pip wheel setuptools &&\
    /venv/bin/pip install dist/*.whl

CMD gunicorn -c yoga_app/gunicorn.conf.py yoga_app.main:app

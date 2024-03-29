FROM maiv/handsigndetect:latest

LABEL Antoine Vignon <vignonantoinem@gmail.com>

WORKDIR /project
RUN pip install flask transformers torch mediapipe streamlit

COPY ./index.html /project
COPY ./sign-language-detect-mediapipe.ipynb /project/notebooks/
COPY ./sign-language-detection.ipynb /project/notebooks/
COPY ./server.py /project
COPY ./helpers.py /project
COPY ./webapp.py /project
COPY ./README.md /project

EXPOSE 5000

CMD python3.9 server.py

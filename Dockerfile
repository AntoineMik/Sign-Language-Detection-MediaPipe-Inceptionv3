FROM maiv/handsigndetect:latest

LABEL Antoine Vignon <vignonantoinem@gmail.com>

WORKDIR /project
RUN pip install flask transformers torch mediapipe

COPY ./index.html /project
COPY ./sign-language-detect-mediapipe.ipynb /project/notebooks/
COPY ./sign-language-detection.ipynb /project/notebooks/
COPY ./server.py /project
COPY ./helpers.py /project
COPY ./webapp.py /project



#Let's define this parameter to install jupyter lab 
# notetebook command so we don't have to use it when running the container 
# with the option -e
ENV JUPYTER_ENABLE_LAB=yes

CMD ["python3", "project.py"]
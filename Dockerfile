FROM jupyter/tensorflow-notebook:latest

LABEL Antoine Vignon <vignonantoinem@gmail.com>

WORKDIR /project

COPY ./index.html /project
COPY ./sign-language-detect-mediapipe.ipynb /project/notebooks/
COPY ./sign-language-detection.ipynb /project/notebooks/


#Let's define this parameter to install jupyter lab 
# notetebook command so we don't have to use it when running the container 
# with the option -e
ENV JUPYTER_ENABLE_LAB=yes

CMD ["jupyter-lab","--ip=0.0.0.0","--no-browser","--allow-root"]
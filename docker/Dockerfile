# Specify base image
#FROM nvcr.io/nvidia/tensorflow:18.08-py3
FROM nvcr.io/nvidia/tensorflow:19.12-tf1-py3

##
# Proxy Settings
##

ENV http_proxy "http://proxy.itc.kansai-u.ac.jp:8080/"
ENV https_proxy "http://proxy.itc.kansai-u.ac.jp:8080/"

##
# Geographic Settings
##

ENV DEBIAN_FRONTEND=noninteractive
RUN sed -i.bak -e "s%http://archive.ubuntu.com/ubuntu/%http://ftp.jaist.ac.jp/pub/Linux/ubuntu/%g" /etc/apt/sources.list
ENV TZ Asia/Tokyo
RUN apt-get update \
  && apt-get install -y tzdata \
  && rm -rf /var/lib/apt/lists/* \
  && echo "${TZ}" > /etc/timezone \
  && rm /etc/localtime \
  && ln -s /usr/share/zoneinfo/Asia/Tokyo /etc/localtime \
  && dpkg-reconfigure -f noninteractive tzdata
  
##
# Install ImageMagick
##
RUN apt-get update && apt-get install -y --no-install-recommends \
        imagemagick \
      && \
    rm -rf /var/lib/apt/lists/


##
# Install dependencies
##
# RUN pip install --upgrade pip
RUN pip install flask

##
# Set User
##
ENV USER student
ENV UID 1001
ENV HOME /home/${USER}
ENV SHELL /bin/bash

RUN groupadd -g ${UID} ${USER}
RUN useradd -g ${UID} -u ${UID} -m -s /bin/bash ${USER}
RUN gpasswd -a ${USER} sudo
RUN echo "${USER}:yoshida" | chpasswd

USER ${USER}
WORKDIR /home/${USER}/

EXPOSE 5000
EXPOSE 6006
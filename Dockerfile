FROM ubuntu:18.04

RUN apt-get update && apt-get install -y \
  wget=1.19.4-1ubuntu2.2 \
  g++=4:7.4.0-1ubuntu2.3 \
  make=4.1-9.1ubuntu1 \
  libfann-dev=2.2.0+ds-3 \
  python3-pip=9.0.1-2.3~ubuntu1.18.04.4 \
  doxygen=1.8.13-10 \
  && rm -rf /var/lib/apt/lists/*
  
RUN pip3 install exhale==0.2.3 sphinx_rtd_theme==0.5.2

RUN mkdir /usr/include/catch2/ && wget https://github.com/catchorg/Catch2/releases/download/v2.12.1/catch.hpp -P /usr/include/catch2/

WORKDIR /home

ENTRYPOINT ["make"]
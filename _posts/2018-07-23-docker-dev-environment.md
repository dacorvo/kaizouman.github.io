---
layout: post
title: 'Use a docker build container for development'
author: 'David Corvoysier'
date: '2018-07-23 16:51:00'
categories:
- Development
tags:
- docker
type: post
---
Using docker containers to meet specific application requirements is now common place: 
instead of having to install many packages on your host, possibly conflicting with already
existing ones, you simply fetch a docker image of the application and run it from there. 

It is less frequent to use docker containers as development sandboxes, but they are however
perfectly suited for the job.

<!--more-->

## Why docker ?

As a versatile developer, I am dealing with multiple different build environments, from
complex cross-compilation setups involving Yocto or buildroot to simpler python or nodejs separated
environments to name a few.

I like to keep my host machine tidy, so there is no way I will install all these environments
without a minimum sandboxing. Anyway, on my typical debian-based development machine, it is
typcially not possible to have conflicting versions of packages to coexist (and keep your sanity).

For years I have been using chroots as sandbox on my host system, but the setup is a bit tedious.
Note: For python development, virtualenv are also a good answer, but for node I don't know of any alternative.

Anyway, since a couple of years, I have been using docker for these sandboxes, for two reasons:

- they are incredibly easy to share and replicate, allowing the whole development team to have the exact same environment,
- there are a lot of base images that you can use.

## Setting up a build container

First, you need a base image for your build environment: you either take an existing image, or you create one.

Then, you can just instantiate your container, mapping your source tree to a convenient directory inside the container.

In the example below, I have for instance mapped a local directory containing jupyter notebooks to the tensorflow
base image root jupyter directory:

```shell
$ docker run -it -v $(pwd):/notebooks -p 8888:8888 tensorflow/tensorflow
```

This will launch the jupyter server, allowing me to edit the notebooks under my local directory, and having
the changes reflected directly in my source directory on the host, and not in the container.

Another example where I map a Yocto tree into an image where I have all Yocto prerequisites, and where an empty /yocto directory exists:

Dockerfile:
```shell
FROM ubuntu:trusty
# This will prevent some errors on the console when installing packages
ARG DEBIAN_FRONTEND=noninteractive
# Jethro build requirements from:
#  http://www.yoctoproject.org/docs/2.0/ref-manual/ref-manual.html#ubuntu-packages
RUN apt-get --quiet update --yes && apt-get --quiet install --yes \
        gawk \
        wget \
        git-core \
        diffstat \
        unzip \
        texinfo \
        gcc-multilib \
        build-essential \
        chrpath \
        socat \
        libsdl1.2-dev \
        xterm \
        && rm -rf /var/lib/apt/lists/*

# Start at the mountpoint
WORKDIR /yocto
# Always source Yocto script when launching a container
CMD  ["/bin/bash", "-c", "source poky/oe-init-build-env"]
```

```shell
$ docker run -it $(pwd):/yocto kaizou/yocto
```

Once the container has been launched, I just have to issue bitbake commands, since the Yocto environment script is launched automatically.

## Avoiding having files owned by root in your development tree

The main drawback of the method described above is that files modified from inside the container may be owned by root,
or worse by some unknown user id.

The trick is to create a special user in the container whose id can be changed during the container invocation using the 
gosu utility.

Below are the corresponding Dockerfile and init script.

Dockerfile:

```shell
from ubuntu
# As root user, install gosu and set up an entrypoint to be able to change the
# default user UID when entering the container
USER root
# This will prevent some errors on the console when installing packages
ARG DEBIAN_FRONTEND=noninteractive
# Install curl
RUN apt-get update && apt-get -y --no-install-recommends install curl
# Install gosu
RUN curl -o /usr/local/bin/gosu -SL "https://github.com/tianon/gosu/releases/download/1.10/gosu-$(dpkg --print-architecture | awk -F- '{ print $NF'})" \
    && chmod +x /usr/local/bin/gosu
# Define the entrypoint
COPY open_session_with_host_uid.sh /open_session_with_host_uid.sh
ENTRYPOINT ["/open_session_with_host_uid.sh"]

# Configure default user environment
ARG UNAME=devel
USER $UNAME

# Switch back to root user to execute the entrypoint script, as we may need
# to change the UID of the default user
USER root
```
open_session_with_host_uid.sh:
```shell
#!/bin/sh
if [ -n "${USER_ID}" ]; then
    usermod -u ${USER_ID} devel;
fi
ARGS=${@:-/bin/bash}
/usr/local/bin/gosu renault "${ARGS}"
```

Just build the image:

```shell
$ docker build . -t kaizou/myimage
```

You can then instantiate a container, passing the current user id on the command line:

```shell
$ docker run -it -v $(pwd):/src -e USER_ID=`id -u` kaizou/myimage
```

## A few words about proxies

In corporate environments, you are often behind a proxy. Docker works behind proxies, but you often need
your guest inside docker to use proxies too.

Below a list of configurations I have experienced:

### Building an image

If your Dockerfile contains commands that need to get access to the outside world, you need to specify the proxy
environments variables as build arguments:

```shell
$ docker build --build-arg http_proxy=${http_proxy} --build-arg https_proxy=${https_proxy} .
```

### Running a container behind a global proxy

Just share the host proxy environment variables with your guest (no specific configuration on the guest required, most of the time):

```shell
$ docker run -it -e http_proxy=${http_proxy} -e https_proxy=${https_proxy} ubuntu:trusty
```

### Running a container behind a local proxy (tricky)

This is a rare configuration, but since I was unlucky enough to experience it ...
Imagine that you have on your host a local http proxy that acts as an authentication agent, and that this proxy
is bound to localhost, and not your real IP (ie http_proxy=http://localhost:3128 for instance).
In that case, if you just export the proxy variable to your guest, it will fail.

The only solution I came up with was to use the host network from docker, instead of using NAT:

```shell
$ docker run -it --net host -e http_proxy=${http_proxy} -e https_proxy=${https_proxy} ubuntu:trusty
```

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

- they are incredibly easy to share and replicate, allowing the whole developemnt team to have the exact same environment,
- there is a lot of base images that you can use.

## Setting up a build container

First, you need a base image for your build environment: you either take an existing image, or you create one.

Then, you can just instantiate your container, mapping your source tree to a convenient directory inside the container.

In the example below, I have for instance mapped a local directory containing jupyter notebooks to the tensorflow
base image root jupyter directory:

~~~~
$ docker run -it -v $(pwd):/notebooks -p 8888:8888 tensorflow/tensorflow
~~~~

This will launch the jupyter server, allowing me to edit the notebooks under my local directory, and having
the changes reflected directly in my source directory on the host, and not in the container.

Another example where I map a Yocto tree into an image where I have all Yocto prerequisites, and where an empty /src directory exists:

~~~~
$ docker run -it $(pwd):/src kaizou/yocto
~~~~

Once the container has been launched, I can cd to the /src directory, source the Yocto environment script and
start issueing bitbake commands.

## Avoiding having files owned by root in your development tree

The main drawback of the method described above is that files modified from inside the container may be owned by root,
or worse by some unknown user id.

The trick is to create a special user in the container whose id can be changed during the container invocation using the 
gosu utility.

Below are the corresponding Dockerfile and init script.

Dockerfile:

~~~~
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
~~~~
open_session_with_host_uid.sh:
~~~~
#!/bin/sh
if [ -n "${USER_ID}" ]; then
    usermod -u ${USER_ID} devel;
fi
ARGS=${@:-/bin/bash}
/usr/local/bin/gosu renault "${ARGS}"
~~~~

Just build the image:

~~~~
$ docker build . -t kaizou/myimage
~~~~

You can then instantiate a container, passing the current user id on the command line:

~~~~
$ docker run -it -v $(pwd):/src -e USER_ID=`id -u` kaizou/myimage
~~~~

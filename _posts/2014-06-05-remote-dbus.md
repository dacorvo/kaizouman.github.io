---
layout: post
title: 'Remote debugging a device running DBus'
author: 'David Corvoysier'
date: '2014-06-05 22:00:00'
categories:
- Linux
tags:
- DBus
- Remote
type: post
redirect_from: /2014/06/remote-dbus/
---
[D-Bus](http://www.freedesktop.org/wiki/Software/dbus/) is the application Bus used on Linux desktop, and is sometimes used on other devices running Linux due to the lack of good alternatives (Android is a good example of D-Bus being adopted by chance).

One of the key benefits of using DBus is that you can issue D-Bus requests from the command-line or from very simple python scripts, which is great for testing.

Trouble is: what if you don't have python on the device, or even worse, if you don't have access to the command-line ?

<!--more-->

The solution is simply to configure D-Bus for remote access.

#Pre-requisites

It is assumed that you have access to the D-Bus system and/or session configuration (typically contained in `/etc/dbus-1/system.conf` and `/etc/dbus-1/session.conf` respectively).

If you don't, but are nevertheless able to add commands to the system `init`, then you might consider creating your own bus:

~~~
export DBUS_SESSION_BUS_ADDRESS=$(dbus-daemon                       \
            --fork                                                  \
            --print-address                                         \
            --config-file=${path-to-your-own-bus-configuration})
~~~

>With a `<listen>` directive similar to this one:
>~~~
><listen>unix:path=/var/run/dbus/orange_bus_socket</listen>
>~~~

#Insert a secondary `<listen>` directive

In the D-Bus configuration file, look for the `<listen>` tag that declares the local port on which the D-Bus deamon listens for incoming requests.

For the system bus, it would be something like:

~~~
<listen>unix:path=/var/run/dbus/system_bus_socket</listen>
~~~

And for the session bus:

~~~
<listen>unix:tmpdir=/tmp</listen>
~~~

Now, insert the following three lines __BEFORE__ the original `<listen>` tag:

~~~
<listen>tcp:host=localhost,bind=*,port=55555,family=ipv4</listen>
<auth>ANONYMOUS</auth>
<allow_anonymous/>
<listen>unix:tmpdir=/tmp</listen>
~~~

Then, start/restart the D-Bus.

#Remote requests using dbus-send

Once your device has been properly configured, you can send D-Bus requests from the command line on your host just by specifying the target IP address and port for the D-Bus deamon:

~~~
export DBUS_SESSION_ADDRESS="tcp:host=localhost,bind=*,port=55555,family=ipv4"
~~~

>Note that it doesn't matter if you want to address the system or session bus:
>just use the same variable.

then use [__dbus-send__](http://dbus.freedesktop.org/doc/dbus-send.1.html) just like you would do when targeting your host.

For instance, the following command will list all services declared on the remote bus:

~~~
dbus-send --session           \
  --dest=org.freedesktop.DBus \
  --type=method_call          \
  --print-reply               \
  /org/freedesktop/DBus       \
  org.freedesktop.DBus.ListNames
~~~

And the same goes for python scripts you would want to run.

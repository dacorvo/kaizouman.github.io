---
layout: post
title: 'Introspecting D-Bus from the command-line'
author: 'David Corvoysier'
date: '2014-06-01 22:00:00'
categories:
- Linux
tags:
- DBus
type: post
redirect_from: /2014/06/dbus-command-line/
---
One of the cool feature of [D-Bus](http://www.freedesktop.org/wiki/Software/dbus/), the Linux desktop application bus, is that it supports introspection.

Even better, you can issue D-Bus introspection requests from the command-line.

<!--more-->

#List D-Bus available services

The following command:

~~~
dbus-send --session           \
  --dest=org.freedesktop.DBus \
  --type=method_call          \
  --print-reply               \
  /org/freedesktop/DBus       \
  org.freedesktop.DBus.ListNames
~~~

will return the list of services available on the session bus

~~~
method return sender=org.freedesktop.DBus -> dest=:1.239 reply_serial=2
   array [
      string "org.freedesktop.DBus"
      string ":1.128"
      string "com.canonical.Unity.Launcher"
      string "org.freedesktop.Notifications"
...
      string "org.freedesktop.IBus"
      string ":1.6"
      string "com.canonical.indicator.power"
   ]
~~~

>Just omit --session to obtain the list of services on the system bus

#Introspect a particular service

All D-Bus service implement the `org.freedesktop.DBus.Introspectable.Introspect` method designed for introspection. 

However, to invoke this method on a service, you need its name __and__ the path at which it has been registered, which is not readily available from the command-line.

The path nevertheless often follows the convention to convert `.` to `/` from the object name, ie `org.freedesktop.DBus` -> `/org/freedesktop/DBus`.

The following command:

~~~
dbus-send --session           \
  --dest=org.freedesktop.DBus \
  --type=method_call          \
  --print-reply               \
  /org/freedesktop/DBus       \
  org.freedesktop.DBus.Introspectable.Introspect
~~~

Returns the interfaces exposed by the org.freedesktop.DBus service

~~~
method return sender=org.freedesktop.DBus -> dest=:1.240 reply_serial=2
   string "<!DOCTYPE node PUBLIC "-//freedesktop//DTD D-BUS Object Introspection 1.0//EN"
   "http://www.freedesktop.org/standards/dbus/1.0/introspect.dtd">
<node>
  <interface name="org.freedesktop.DBus">
    <method name="Hello">
      <arg direction="out" type="s"/>
    </method>
    <method name="RequestName">
      <arg direction="in" type="s"/>
      <arg direction="in" type="u"/>
      <arg direction="out" type="u"/>
    </method>
    <method name="ReleaseName">
      <arg direction="in" type="s"/>
      <arg direction="out" type="u"/>
    </method>
 ...
    <method name="ReloadConfig">
    </method>
    <method name="GetId">
      <arg direction="out" type="s"/>
    </method>
    <signal name="NameOwnerChanged">
      <arg type="s"/>
      <arg type="s"/>
      <arg type="s"/>
    </signal>
    <signal name="NameLost">
      <arg type="s"/>
    </signal>
    <signal name="NameAcquired">
      <arg type="s"/>
    </signal>
  </interface>
  <interface name="org.freedesktop.DBus.Introspectable">
    <method name="Introspect">
      <arg direction="out" type="s"/>
    </method>
  </interface>
</node>
~~~

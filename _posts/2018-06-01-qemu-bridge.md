---
layout: post
title: 'Bridging two QEmu guests'
author: 'David Corvoysier'
date: '2018-06-01 10:38:00'
categories:
- Development
tags:
- qemu
- bridge
type: post
---
QEmu is not only useful to test code for another platform on your host, it also allows launching guest system images.

This is extremely useful to the embedded developers, as it allows to perform integration tests in a virtual environment.

You can even simulate a platform combining several network nodes thanks to the QEmu bridging capabilities, and in this

article I will explain how to establish IP communication between two guests on the same host.

<!--more-->

The first step is to create the network bridge allowing your QEmu instances to share the same network.

```shell
$ ip link add name br0 type bridge
$ ip link set dev br0 up
```
Then, you need to tell QEmu that the bridge is legit by adding it to the QEmu 'ACL':

```shell
$ echo 'allow br0' | sudo tee -a /etc/qemu/bridge.conf
```
Specifying the bridge netdev option will tell QEmu to create virtual TAP interfaces for each image and add them to the bridge.
We will force the IP addresses to the expected values on the QEmu command line, and assign different MAC addresses 
to allow the bridge to differentiate between the two images for ethernet routing.

```shell
$ sudo qemu-system-i386 \
        -kernel ${KERNEL_IMAGE} \
        -append "root=/dev/vda ip=192.168.7.1::192.168.7.0:255.255.255.0 console=ttyS0" \
        -drive file=${ROOTFS_IMAGE},if=virtio,format=raw \
        -netdev bridge,id=hn1 -device virtio-net,netdev=hn1,mac=52:54:00:12:34:50
$ sudo qemu-system-i386 \
        -kernel ${KERNEL_IMAGE} \
        -append "root=/dev/vda ip=192.168.7.2::192.168.7.0:255.255.255.0 console=ttyS0" \
        -drive file=${ROOTFS_IMAGE},if=virtio,format=raw \
        -netdev bridge,id=hn1 -device virtio-net,netdev=hn1,mac=52:54:00:12:34:51
```
You can verify your network setup, to check that the tap interfaces are indeed added to the bridge:

```shell
$ip a
...
8: br0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc noqueue state UP group default qlen 1000
    link/ether fe:11:5d:e1:a1:95 brd ff:ff:ff:ff:ff:ff
    inet6 fe80::4c1c:87ff:fe8f:536f/64 scope link
       valid_lft forever preferred_lft forever
9: tap0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc pfifo_fast master br0 state UNKNOWN group default qlen 1000
    link/ether fe:db:60:a1:eb:6b brd ff:ff:ff:ff:ff:ff
10: tap1: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc pfifo_fast master br0 state UNKNOWN group default qlen 1000
    link/ether fe:11:5d:e1:a1:95 brd ff:ff:ff:ff:ff:ff
```

Now, the two containers should see each other. 

Warning: if you have docker installed on your host, it may have created an iptable rule that forbids IP forwarding:
the symptom will be that ARP will work, allowing your containers to see each other, but all IP traffic between your
containers will be blocked.
In that case, you need to disable iptables for network bridges (as root):

```shell
$ echo 0  | sudo tee /proc/sys/net/bridge/bridge-nf-call-iptables
```

You can monitor the IP traffic between the two containers using tshark (or tcpdump):

```shell
$ sudo tshark -i br0
```

The virtual tap interfaces are automatically destroyed when you exit a container (CTRL A+X).

Once done, you can destroy the bridge:

```shell
ip link set dev br0 down
ip link del br0
```

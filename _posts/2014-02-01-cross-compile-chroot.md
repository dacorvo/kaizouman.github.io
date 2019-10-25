---
layout: post
title: 'Safely meet cross-compilation environments requirements using a chroot'
author: 'David Corvoysier'
date: '2014-02-01 22:00:00'
categories:
- Embedded
- Linux
tags:
- cross-compilation
- schroot
type: post
published: true
redirect_from: /2014/02/cross-compile-chroot/
---

It is generally a good practice to cross-compile in an isolated environment to avoid plaguing your host system with unneccessary tooling and dependencies, or worse corrupting it.

Another issue arises when using binary toolchains that have been built for a specific host (typically a 32 bits Linux host, when most users are today using a 64 bits distribution).

These two concerns (isolation and binary compatibility) are well addressed by [chroot environments](https://wiki.debian.org/chroot).

<!--more-->

## Creating a chroot for your build environment

A chroot is basically a special directory on your computer that behaves like another operating system inside your existing operating system.

It is particularly useful when using development SDKs that require a specific system configuration, as it allows to use them "inside" your real system without altering it.

Please refer to the [Ubuntu Chroot Documentation](https://help.ubuntu.com/community/BasicChroot) for more details.

### Pre-requisites

Install the schroot utility

    sudo apt-get install schroot

Install the debootstrap utility

    sudo apt-get install debootstrap

Optionally (recommended), tell schroot to cleanup orphan sessions on system startup

    sudo editor /etc/default/schroot
    ...
    # What do we want to do with "orphan" sessions when starting or
    # restarting?  Recover them (leave empty or set to "recover") or just
    # end them (set to "end")?
    START_ACTION="end"

### Create a chroot for a specific Ubuntu version

1.
Create the chroot directory

    sudo install -d /<path-to-my-chroots>/<mychroot>

2.
Edit the schroot configuration

    sudo editor /etc/schroot/schroot.conf

Append at the end of file:

    [mychroot]
    type=directory
    description=<my-description>
    directory=/<path-to-my-chroots>/<mychroot>
    users=<my-user-login>
    root-groups=root
    # The following line is optional but may prove
    # useful if you want to export proxy settings
    preserve-environment=true
    # The line below only if you are creating a 32
    # bits chroot on a 64 bits host
    personality=linux32


3.
Create the chroot (variant=buildd is to install development tools, arch i386 is for using 32 bits binaries ; ubuntu-version is any of lucid, precise, ...)

    sudo debootstrap --variant=buildd \
                     --arch i386 <ubuntu-version> <path-to-my-chroots>/<mychroot> \
                     http://archive.ubuntu.com/ubuntu/

or

    sudo debootstrap --variant=buildd \
                     --arch amd64 <ubuntu-version> <path-to-my-chroots>/<mychroot> \
                     http://archive.ubuntu.com/ubuntu/

4.
Test your chroot environment

    schroot -c mychroot -- uname -a

If you created a 32 bits chroot, you should see something like

    Linux ... i686 GNU/Linux

Otherwise, you should see

    Linux ... x86_64 x86_64 x86_64 GNU/Linux


5.
Generate locale for your system

    sudo schroot -c mychroot -- locale-gen en_EN.UTF-8

or

    sudo schroot -c mychroot -- locale-gen fr_FR.UTF-8

### Install missing dependencies

Once your chroot has been setup, you will probably need to install a few packages that are not included in the buildd variant. Since this requires administration privileges, you will need to issue commands as root in the chroot.

There are at least three ways to achieve this:

1.
sudo on command line

    sudo schroot -c mychroot -- apt-get update
    sudo schroot -c mychroot -- apt-get install

2.
chroot as root

Enter chroot as root:

    sudo schroot -c mychroot

On chroot prompt:

    apt-get update
    apt-get install ...

3.
sudo in chroot

Install sudo from outside:

    sudo schroot -c mychroot -- apt-get install sudo

Enter chroot with standard privileges:

    schroot -c mychroot

On chroot prompt:

    sudo apt-get update
    sudo apt-get install ...

Note: For alternatives 2 & 3, beware of environment variables like proxy configuration that will not be exported to the sudo environment unless you edit /etc/sudoers accordingly in your chroot.

    Defaults env_keep = "http_proxy https_proxy ftp_proxy"

## Using a chroot to cross-compile for a target

### Issueing commands

There are two ways to issue commands in a chroot environment:

1.
From your host, on a per-command basis

    schroot -c mychroot -- <command>

>The actual syntax is schroot -c mychroot <command>, but inserting '--' makes sure that the arguments between the schroot command and the target <command> are not mixed.

2.
From the chroot itself

    schroot -c mychroot
    (mychroot)$ <command>

The first approach is well suited for inclusion in automation scripts that switch between pure hosts commands and chroot commands. It is a bit clumsy and error-prone for typical command-line usage.

The second approach is the one that you would use when actually developing in a chroot. I personnally keep a shell opened inside the chroot to issue all my build commands.

### Editing files

I do not recommend installing an editor inside you chroot, as it will most likely pull an awful lot of dependencies that may conflict with your build environment.

Instead, I recommend editing files from your host: using a develoment workspace inside your $HOME guarantees that the chroot sees it (thanks to the automatic mount point performed by schroot).


### Bonus tip: orphan sessions manual cleanup

If you have exited abnormally a schroot session, there might be some useless bind mounts left over from the schroot session (/proc & /sys for instance). You can tell schroot to manually clean up those by issueuing 

    schroot --all-sessions --end-session

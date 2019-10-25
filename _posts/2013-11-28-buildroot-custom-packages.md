---
layout: post
title: 'Build customized embedded linux firmwares using buildroot'
author: 'David Corvoysier'
date: '2013-11-28 22:00:00'
categories:
- Embedded
- Linux
tags:
- cross-compilation
- buildroot
type: post
published: true
redirect_from: /2013/11/buildroot-custom-packages/
---

Many embedded devices run customized Linux systems that include a more or less wide range of features on top of a standard kernel delegating proprietary hardware handling to a minimal set of specific drivers.

Chipset vendors typically provide SDKs to build such customized solutions without too much pain, but these proprietary tools often lack the flexibility required when you want ot build a solution that either differs too much from what was expected by the vendor or includes software components that were designed too long after the SDK was released.

<!--more-->

> Example: A lot of chipset vendors SDKs in the set-top-box business still require gstreamer 0.10.x despite the fact that it is no longer supported

Several community-driven projects try to address these issues by providing generic solutions to build embedded systems for a wide range of chipsets. The two biggest communities are built around [buildroot](http://buildroot.org) and [Open Embedded](www.openembedded.org)/[Yocto](https://www.yoctoproject.org/).

I won't go into too much details about these two solutions here, but in a nutshell they both provide a framework to build a system from scratch using a set of build 'recipes', solving cross-compilation issues along the way.
The main difference between the two solutions is that buildroot aims at simpler/smaller ad-hoc solutions whereas Yocto can be used to build full Linux distros. 

Unless you are using only open source software, whatever tool you use, one of the issue you will need to address is how to separate the open source from the proprietary components.

This article describes a neat solution to do this using buildroot.

## About buildroot

From their website:

>Buildroot is a set of Makefiles and patches that makes it easy to generate a complete embedded Linux system. Buildroot can generate any or all of a cross-compilation toolchain, a root filesystem, a kernel image and a bootloader image. Buildroot is useful mainly for people working with small or embedded systems, using various CPU architectures (x86, ARM, MIPS, PowerPC, etc.) : it automates the building process of your embedded system and eases the cross-compilation process.

Links:

* [Buildroot Home page](http://buildroot.org)
* [Buildroot documentation](http://buildroot.org/downloads/manual/manual.html)

So, really, what happens when you create a build environment for your customized system using buildroot is that it generates a root makefile and a package configuration in your target directory (by the way, you also create your build enviroment using `make`).

You can then issue typical make commands on this root makefile to build your firmware according to the specified configuration, the root makefile invoking the specific package makefiles with the appropriate set of variables. See: no magic here, just an automated process based on well-known techniques.

## Buildroot standard setup

First things first, before using buildroot you need to [fetch it](http://buildroot.org/download) and install a few [prerequisites](http://buildroot.org/downloads/manual/manual.html#requirement) (Warning: depending on what you want to do, [your mileage may vary](http://buildroot.org/downloads/manual/manual.html#requirement-optional)).

To setup a build environment, you would then invoke the creation of a blank target using a command like the following:

~~~
 make -C <path-to-buildroot>        \
      O=<path-to-target-directory>  \
      menuconfig
~~~

This will open a curses-based interface allowing you to set the build parameters for your target and select the packages you want to include (more later on the packages). The whole configuration process is explained in details in the [buildroot documentation](http://buildroot.uclibc.org/downloads/manual/manual.html#configure).

Once done, you will probably want to [save your configuration](http://buildroot.uclibc.org/downloads/manual/manual.html#customize-store-buildroot-config) in a buildroot `defconfig` file:

~~~
make -C <path-to-target-directory>     \
     BR2_DEFCONFIG=<path-to-defconfig> \
     savedefconfig
~~~

This `defconfig` file can later be used to recreate the exact same build configuration on another host system:

~~~
make -C <path-to-buildroot>            \
     O=<path-to-target-directory>      \
     BR2_DEFCONFIG=<path-to-defconfig> \
     defconfig
~~~

And ... well that's about it: now you only need to launch make to generate your firmware:

~~~
make -C <path-to-target-directory>
~~~

The output files are located under `<path-to-target-directory>/images`.

## About buildroot packages

A buildroot package is actually a set of metadata describing a source archive and a build "recipe".

The package metadata are described in a `Config.in` file that uses the Linux [KConfig](https://www.kernel.org/doc/Documentation/kbuild/kconfig-language.txt) syntax.

The `Config.in` is processed by buildroot when the `make menuconfig` command is invoked to:

* make the package available in the configuration user interface if its prerequisites are satisfied,
* provide its name and description,
* identify the dependencies that need to be added when it is selected.

The package recipe is described in a `.mk` file matching the package name that uses the [Make](http://www.gnu.org/software/make/manual/make.html) syntax.

The `<package>.mk` file is processed by buildroot when the `make <package>` command is invoked to:

* fetch the package source code from a remote or local repository (several methods are supported including tarballs, subversion, git or mercurial),
* prepare the sources (copy, untar and patch),
* configure the build,
* build the package,
* install it to the staging (for other dependent packages to use it during their build),
* install it to the target.

The complete and exhaustive documentation on adding new packages is available in the [Buildroot manual](http://buildroot.net/downloads/manual/manual.html#adding-packages).

## Adding external packages to buildroot

Until quite recently, buildroot was only aware of packages located under the `<path-to-buildroot>/packages` directory (this is were the community-contributed packages are), which means that in order to get your custom packages available to buildroot you had to put it there, which caused all sorts of problems if you wanted to keep it in a separate configuration management system.

Thanks to the new [BR2_EXTERNAL](http://buildroot.uclibc.org/downloads/manual/manual.html#outside-br-custom) mechanism, it is now allowed to store package definitions and even custom configuration outside of the buildroot directory.

### Prepare a BR2_EXTERNAL directory

The whole point of this new approach is to allow the cutomized (and often closed-source) part of your system to live in a dedicated project repository.

This is how a typical `BR2_EXTERNAL` compatible project directory looks like:

~~~
+-- Config.in
+-- external.mk
+-- configs/
|   +-- <boardname>_defconfig
|
+-- package/
    +-- <company>/
        +-- package1/
        |    +-- Config.in
        |    +-- package1.mk
    +-- package2/
        +-- Config.in
        +-- package2.mk
~~~

The `Config.in` file is the root of the 'external' packages tree, and must include all `Config.in` files describing each individual package.

~~~
source "$BR2_EXTERNAL/package/<company>/package1/Config.in"
source "$BR2_EXTERNAL/package/package2/Config.in"
~~~

The list of external packages is available under the 'User-provided options' menu item in the buildroot user interface.

In the same way, the `external.mk` file is the top makefile for external packages, and must include all the corresponding makefiles:

~~~
include $(sort $(wildcard $(BR2_EXTERNAL)/package/*/*.mk))
include $(sort $(wildcard $(BR2_EXTERNAL)/package/*/*/*.mk))
~~~

The `$(BR2_EXTERNAL)/configs` directory is automatically recognized by buildroot and will be searched for saved defconfig files.

### Creating a new external package

Basically, adding a new package to your project involves four steps:

* create a new `$(BR2_EXTERNAL)/package/<my-package>` directory,
* create `package/<my-package>/Config.in`,
* create `package/<my-package>/<my-package>.mk`,
* reference the new package in `$(BR2_EXTERNAL)/Config.in`:

~~~~
source "$BR2_EXTERNAL/package/<my-package>/Config.in"
~~~~

The new package can now be added to your target by selecting it in the configuration interface, under the 'User-provided options' menu item.

### Configuring buildroot to use a BR2_EXTERNAL directory

The path to the `BR2_EXTERNAL` has to be specified when creating the target directory, but it can be omitted in later commands:

~~~ 
make -C <path-to-buildroot>            \
     O=<path-to-target-directory>      \
     BR2_EXTERNAL=<path-to-externals>  \
     <target>_defconfig
~~~

Where `<target>_defconfig` is a buildroot defconfig file located under `$(BR2_EXTERNAL)/configs`.

As recommended in the buildroot documentation, adding a `$(BR2_EXTERNAL)/board directory` to store board specific configuration is generally a good idea:

~~~
+-- Config.in
+-- external.mk
...
+-- board/
|       +-- <boardname>/
|           +-- linux.config
|           +-- busybox.config
|           +-- <other configuration files>
|           +-- post_build.sh
|           +-- post_image.sh
|           +-- rootfs_overlay/
|               +-- etc/
|               +-- <some file>
~~~

The `$(BR2_EXTERNAL)/boards` directory is not automatic recognized by buildroot, so you will need to reference it directly in the defconfig:

~~~
BR2_ROOTFS_OVERLAY=$(BR2_EXTERNAL)/board/<boardname>/overlay/
BR2_LINUX_KERNEL_CUSTOM_CONFIG_FILE=$(BR2_EXTERNAL)/board/<boardname>/kernel.config
...
~~~

I personnally also find interesting to specify a global patch directory in the `external.mk`, with patches that are applied regardless of the target configuration selected:

~~~
external.mk:
BR2_GLOBAL_PATCH_DIR="$(BR2_EXTERNAL)/patches"
~~~

~~~
+-- Config.in
+-- external.mk
|
...
|
+-- patches/
    +-- foo/
    |   +-- <some patch>
    +-- libbar/
        +-- <some other patches>
~~~

### Generate the customized firmware

There is absolutely nothing new here: you will use the exact same command to generate your firmware: 

~~~
make -C <path-to-target-directory>
~~~

And the output files will still be located under `<path-to-target-directory>/images`.

## The icing on the cake: control buildroot version using a submodule

When using `$(BR2_EXTERNAL)`, you lose the ability to control which version of buildroot is used to build your system.

This can be overcome easily if your project uses git by adding buildroot itself as a submodule in your project repository.

~~~
cd $(BR2_EXTERNAL)
git submodule add git://git.buildroot.net/buildroot buildroot
git submodule init
git checkout <whatever-version-you-want>
git add buildroot
git commit -m 'Now I have selected the buildroot version I want'
~~~

So, when someones checks out your project, it can automatically fetch the proper buildroot version:

~~~
git checkout <project git-repo>
git submodule init
git submodule update
~~~

neat, uh ?

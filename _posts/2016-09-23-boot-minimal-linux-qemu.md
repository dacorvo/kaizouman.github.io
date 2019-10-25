---
layout: post
title: 'Build and boot a minimal Linux system with qemu'
author: 'David Corvoysier'
date: '2016-09-23 16:00:00'
categories:
- System
tags:
- build
- linux
- croostool-ng
- qemu
- busybox
type: post
redirect_from: /2016/09/boot-minimal-linux-qemu/
---

When you want to build a Linux system for an embedded target these days, it is very unlikely that you decide to do it from scratch.

Embedded Linux build systems are really smart and efficients, and will fit almost all use cases: should you need only a simple system, [buildroot](https://buildroot.org/) should be your first choice, and if you want to include more advanced features, or even create a full distribution, [Yocto](https://www.yoctoproject.org/) is the way to go.

That said, even if these tools will do all the heavy-lifting for you, they are not perfect, and if you are using less common configurations, you may stumble upon issues that were not expected. In that case, it may be important to understand what happens behind the scenes.

In this post, I will describe step-by-step how you can build a minimal Linux system for an embedded target and boot it using [QEMU](http://wiki.qemu.org/Main_Page).

<!--more-->

# Install QEMU

[QEMU](http://wiki.qemu.org/Main_Page) is available for all major distros.

```shell
sudo apt-get install qemu
```

In this post I will create a system for an ARM target, just to make sure I don't mix between my host and target systems (see the last paragraph of [this introduction on cross-compilation](https://landley.net/writing/docs/cross-compiling.html)).

You can list the ARM machines your [QEMU](http://wiki.qemu.org/Main_Page) setup supports from the command-line:

```shell
$ qemu-system-arm --machine help
Supported machines are:
versatileab          ARM Versatile/AB (ARM926EJ-S)
...
mainstone            Mainstone II (PXA27x)
...
midway               Calxeda Midway (ECX-2000)
virt                 ARM Virtual Machine
borzoi               Borzoi PDA (PXA270)
```

I will use in this tutorial an old Intel ARM platform, the Mainstone.

>The only reason I chose this platform is because the maintainer of this board is Robert Jarzmik, who has been sitting next to me in the Open space for the last year. He is _very_knowledgeable on the Kernel and also very nice. Thanks to you, Bob !

#Generate the toolchain

To generate the binaries for our embedded target, we need a toolchain, which is a set of tools targeting the corresponding processor architecture.

Most of the time, the board manufacturer will have provided the toolchain as part of the BSP (Board Support Package).

Generating a toolchain used to be quite painful, but since the awesome [crosstool-ng](http://crosstool-ng.org/) tool has been made available, this is a piece of cake.

>More namedropping: kudos to my friend [Yann E. Morin](http://ymorin.is-a-geek.org/) for developping [crosstool-ng](http://crosstool-ng.org/)

First, we need to fetch and install the tool.

```shell
$ wget http://crosstool-ng.org/download/crosstool-ng/crosstool-ng-1.22.0.tar.xz
$ tar xf crosstool-ng-1.22.0.tar.xz
$ cd crosstool-ng/
$ ./configure
$ make
$ sudo make install
```

You can list the pre-configured toolchains that your cross-tool ng version supports:

```shell
$ ct-ng list-samples
Status  Sample name
  LN    config
  MKDIR config.gen
  IN    config.gen/arch.in
  IN    config.gen/kernel.in
  IN    config.gen/cc.in
  IN    config.gen/binutils.in
  IN    config.gen/libc.in
  IN    config.gen/debug.in
[G..]   alphaev56-unknown-linux-gnu
...
[G..]   armeb-unknown-linux-uclibcgnueabi
...
[G..]   xtensa-unknown-linux-uclibc
 L (Local)       : sample was found in current directory
 G (Global)      : sample was installed with crosstool-NG
 X (EXPERIMENTAL): sample may use EXPERIMENTAL features
 B (BROKEN)      : sample is currently broken
```

For the Mainstone board, we will use a generic ARM toolchain with [uCLibc](https://www.uclibc.org/), a smaller C library for embedded targets.

You can get the details of the toolchain that will be produced from the command-line:

```shell
$ ct-ng show-arm-unknown-linux-uclibcgnueabi
  IN    config.gen/arch.in
  IN    config.gen/kernel.in
  IN    config.gen/cc.in
  IN    config.gen/binutils.in
  IN    config.gen/libc.in
[G..]   arm-unknown-linux-uclibcgnueabi
    OS             : linux-4.3
    Companion libs : gmp-6.0.0a mpfr-3.1.3 mpc-1.0.3 libelf-0.8.13 expat-2.1.0 ncurses-6.0
    binutils       : binutils-2.25.1
    C compilers    : gcc  |  5.2.0
    Languages      : C,C++
    C library      : uClibc-ng-1.0.9 (threads: nptl)
    Tools          : dmalloc-5.5.2 duma-2_5_15 gdb-7.10 ltrace-0.7.3 strace-4.10
```

Let's generate (this _will_ take a while):

```shell
$ ct-ng arm-unknown-linux-uclibcgnueabi
$ ct-ng build
```

By default, the toolchain will be installed under $(HOME)/x-tools/arm-unknown-linux-uclibcgnueabi. In order to use it, we add the toolchain bin directory to the PATH:

```shell
$ export PATH="${PATH}:${HOME}/x-tools/arm-unknown-linux-gnueabi/bin"
$ arm-unknown-linux-uclibcgnueabi-gcc --version
arm-unknown-linux-uclibcgnueabi-gcc (crosstool-NG crosstool-ng-1.22.0) 5.2.0
Copyright (C) 2015 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```

>Note that I have added a small routine to my shell startup script to automatically add paths to toolchains:
>```shell
>for dir in `ls ~/x-tools`; do
>PATH=~/x-tools/$dir/bin:$PATH
>done
>export PATH
>```

# Sanity check: test cross-compilation environment

It is always a good practice to verify at regular intervals that your setup is correct. Here, we will make sure that the toolchain is able to generate ARM code that can be run by qemu-arm, the QEMU ARM CPU emulator.

For those unfamiliar with cross-compilation, this may also help to put things in perspective.

We will compile a very simple program:

```
main.c

#include <stdio.h>

int main(int argc, char*argv[])
{
	printf("Genuinely generated by the toolchain\n");
}
```

Let's first build with a naive command:

```shell
$ arm-unknown-linux-uclibcgnueabi-gcc main.c -o sanity
$ chmod +x sanity
```

We verify that sanity is an ARM exec that cannot run on our system:

```shell
$ ./sanity
bash: ./sanity: cannot execute binary file: Exec format error
$ file sanity
sanity: ELF 32-bit LSB  executable, ARM, EABI5 version 1 (SYSV), dynamically linked (uses shared libs), not stripped
```

Now, let's try to run it with QEMU:

```shell
$ qemu-arm sanity
/lib/ld-uClibc.so.0: No such file or directory
```

What happened ? The reason we get this error is because by default GCC has generated a sanity executable that requires dynamic linking of system libraries, as the following command reveals:

```shell
$ readelf -d sanity

Dynamic section at offset 0x4f0 contains 18 entries:
  Tag        Type                         Name/Value
 0x00000001 (NEEDED)                     Shared library: [libc.so.1]
 0x0000000c (INIT)                       0x102d4
...
```

Here, QEMU needs to find the C library, and to load it using the dynamic linker, which happens to be also a library, ld-uClibc.so.0, as the INTERP program header reveals:

```shell
$ readelf -l sanity
Elf file type is EXEC (Executable file)
Entry point 0x10334
There are 6 program headers, starting at offset 52

Program Headers:
  Type           Offset   VirtAddr   PhysAddr   FileSiz MemSiz  Flg Align
  PHDR           0x000034 0x00010034 0x00010034 0x000c0 0x000c0 R E 0x4
  INTERP         0x0000f4 0x000100f4 0x000100f4 0x00014 0x00014 R   0x1
      [Requesting program interpreter: /lib/ld-uClibc.so.0]
...
```

Both libraries are under the toolchain 'sysroot' directory.

>Should you decide to support dynamic linking, the dynamic linker and the C library should at some point end up on your target.

Specifically for that purpose, QEMU supports specifying the path to dynamically linked libraries using the -L option or the QEMU_LD_PREFIX environment variable.

```shell
$ qemu-arm -L ~/x-tools/arm-unknown-linux-uclibcgnueabi/arm-unknown-linux-uclibcgnueabi/sysroot/ sanity
Genuinely generated by the toolchain
```

or

```shell
$ QEMU_LD_PREFIX=~/x-tools/arm-unknown-linux-uclibcgnueabi/arm-unknown-linux-uclibcgnueabi/sysroot/ qemu-arm sanity
Genuinely generated by the toolchain
```

If you want to avoid these linking issues, you can tell GCC to generate a static executable instead:

```shell
$ arm-unknown-linux-uclibcgnueabi-gcc -static main.c -o sanity
$ qemu-arm sanity
Genuinely generated by the toolchain
```

# Configure and build the Linux Kernel

At the time this article is written, the latest Kernel stable version is 4.7.5.

```shell
$ mkdir linux
$ wget https://cdn.kernel.org/pub/linux/kernel/linux-4.7.5.tar.xz -O linux/linux-4.7.5.tar.xz
$ tar xf linux/linux-4.7.5.tar.xz -C linux
```

We select the mainstone configuration to build the Kernel

```shell
$ make -C linux/linux-4.7.5 ARCH=arm mainstone_defconfig O=linux/build
```

>You need to specify the architecture to tell the Kernel where it should look for existing configurations (here arch/arm/configs)

The Linux Kernel is very versatile in the way it boots, and it can be frankly overwhelming if you consider all options.

In this article, I will illustrate two boot modes: a stand-alone Kernel with a RAM initrd, and a Kernel that boots on a root filesystem on an SD card.

As per the Linux Kernel documentation:

>initrd provides the capability to load a RAM disk by the boot loader.
This RAM disk can then be mounted as the root file system and programs
can be run from it. Afterwards, a new root file system can be mounted
from a different device. The previous root (from initrd) is then moved
to a directory and can be subsequently unmounted.
>
>initrd is mainly designed to allow system startup to occur in two phases,
where the kernel comes up with a minimum set of compiled-in drivers, and
where additional modules are loaded from initrd.

initrd is primarily intended to be a bootstrap in RAM that allows the Kernel to get access to the 'real' rootfs, but we can also use it to simply boot the Kernel without providing a rootfs.

We will see how we can create an initrd in the subsequent paragraphs.

The mainstone default configuration is fairly minimal, and we will need to add a few options to support these two boot modes.

First, we need to add initrd support by activating the BLK_DEV_INITRD configuration option.

Second, we need to add SD cards support for the mainstone board, that belongs to the PXA family. The driver is called MultiMedia card driver for PXA, and it requires Direct Memory Access: we will therefore need to select MMC, MMC_PXA, DMADEVICES and PXA_DMA.

We also need to activate the AEABI configuration to make sure the Kernel uses the latest ARM EABI convention. As per the Linux Kernel documentation:

>This option allows for the kernel to be compiled using the latest ARM ABI (aka EABI).  This is only useful if you are using a user space environment that is also compiled with EABI.

We need to add these options manually using the curses menuconfig interface:

```shell
$ make -C linux/build ARCH=arm menuconfig
```

>General Setup->Initial RAM filesystem and RAM disk (initramfs/initrd) support
>Device Drivers->MMC/SD/SDIO card support->Intel PXA25x/.. Multimedia Card Interface support
>Device Drivers->DMA Engine support->PXA DMA support
>Kernel Features->Use the ARM EABI to compile the kernel

Once our Kernel has been properly configured, we can build it:

```shell
$ make -C linux/build ARCH=arm CROSS_COMPILE=arm-unknown-linux-uclibcgnueabi-
```

At the end of the build, our Kernel will be under linux/build/arch/arm/boot.

```shell
$ ls linux/build/arch/arm/boot/
compressed  Image  zImage
```

# Sanity check: launch the Linux Kernel with QEMU

We verify that the Kernel has been properly generated by launching it with qemu-system-arm, the [QEMU](http://wiki.qemu.org/Main_Page) system emulator (note the difference with qemu-arm, the CPU emulator).

We pass four parameters on the command-line:

- kernel: path to our Kernel,
- machine: the machine w euse (here 'mainstone'),
- serial: set to 'stdio' to the Kernel printk logs in the console,
- append: parameters to add to the Kernel command-line

```shell
$ qemu-system-arm -kernel linux/zImage -serial stdio -append 'console=ttyS0' -M mainstone
Two flash images must be given with the 'pflash' parameter
```

The mainstone board has two 64 Mb flash banks whose images must be provided on the qemu-system-arm command-line.

We create two empty images:

```shell
$ dd if=/dev/zero of=mainstone-flash0.img bs=1024 count=65536
$ dd if=/dev/zero of=mainstone-flash1.img bs=1024 count=65536
```

We can now launch the Kernel.

```shell
$ qemu-system-arm -kernel linux/zImage -append 'console=ttyS0' -machine mainstone -serial stdio -pflash mainstone-flash0.img -pflash mainstone-flash1.img
Booting Linux on physical CPU 0x0
Linux version 4.7.5 (xxx@yyy) (gcc version 5.2.0 (crosstool-NG crosstool-ng-1.22.0) ) #1 Tue Sep 27 09:35:52 CEST 2016
CPU: XScale-PXA270 [69054117] revision 7 (ARMv5TE), cr=00007977
...
XScale iWMMXt coprocessor detected.
VFS: Cannot open root device "(null)" or unknown-block(0,0): error -6
Please append a correct "root=" boot option; here are the available partitions:
Kernel panic - not syncing: VFS: Unable to mount root fs on unknown-block(0,0)
...
```

It still fails because we didn't provide a rootfs nor an initrd.

# Create a tiny init

Let's create a simplistic bootstrap:

```
main.c:

#include <stdio.h>

void main()
{
	printf("Tiny init ...\n");
	while(1);
}
```

We compile it using the ARM toolchain, passing a few CFLAGS to specify the mainstone CPU instruction set:

```shell
$ arm-unknown-linux-uclibcgnueabi-gcc -static -march=armv5te -mtune=xscale -Wa,-mcpu=xscale main.c -o init
$ chmod +x init
```

We will now use that bootstrap to boot the system after the Kernel has been loaded.

# RAM boot using initrd

We create a CPIO RAM image that contains only the init program:

```shell
$ echo init | cpio -o --format=newc > initramfs
```

Now, if we launch the Kernel again, specifying our initramfs, we end up in the tiny init loop:

```shell
$ qemu-system-arm -kernel linux/zImage -append 'console=ttyS0' -machine mainstone -serial stdio -pflash mainstone-flash0.img -pflash mainstone-flash1.img -initrd initramfs
Booting Linux on physical CPU 0x0
Linux version 4.7.5 (xxx@yyy) (gcc version 5.2.0 (crosstool-NG crosstool-ng-1.22.0) ) #1 Tue Sep 27 09:35:52 CEST 2016
CPU: XScale-PXA270 [69054117] revision 7 (ARMv5TE), cr=00007977
...
XScale iWMMXt coprocessor detected.
Freeing unused kernel memory: 148K (c03cf000 - c03f4000)
This architecture does not have kernel memory protection.
Tiny init ...
```

# Boot on a SD card image

We will now create an SD card image containing the tiny init code.

```shell
$ qemu-img create init.img 128K
```

We format the SD card image with an ext2 file-system.

```shell
$ mkfs.ext2 init.img
mke2fs 1.42.13 (17-May-2015)
Discarding device blocks: done
Creating filesystem with 128 1k blocks and 16 inodes

Allocating group tables: done
Writing inode tables: done
Writing superblocks and filesystem accounting information: done
```

Then, we can mount it and copy the init program into the image

```shell
$ mkdir tmp
$ sudo mount -o loop init.img tmp
$ mkdir -p tmp/sbin
$ sudo cp init tmp/sbin/
$ sudo umount tmp
$ rmdir tmp
```

>Note that the Kernel expects the init bootstrap to be under /sbin/init, and not at the root of the file system like in the initram file system.

We can now launch the Kernel specifying that the rootfs is on /dev/mmcblk0, which is the pseudo-device for the SD card passed to [QEMU](http://wiki.qemu.org/Main_Page) with the -sd option.

```shell
$ qemu-system-arm -kernel linux/zImage -append 'console=ttyS0 root=/dev/mmcblk0' -machine mainstone -serial stdio -pflash mainstone-flash0.img -pflash mainstone-flash1.img -sd init.img
Booting Linux on physical CPU 0x0
Linux version 4.7.5 (xxx@yyy) (gcc version 5.2.0 (crosstool-NG crosstool-ng-1.22.0) ) #1 Tue Sep 27 09:35:52 CEST 2016
CPU: XScale-PXA270 [69054117] revision 7 (ARMv5TE), cr=00007977
...
XScale iWMMXt coprocessor detected.
mmc0: host does not support reading read-only switch, assuming write-enable
mmc0: new SD card at address 4567
mmcblk0: mmc0:4567 QEMU! 1.00 GiB
VFS: Mounted root (ext2 filesystem) readonly on device 179:0.
Freeing unused kernel memory: 152K (c03ee000 - c0414000)
This architecture does not have kernel memory protection.
Tiny init ...
```

Voila !

In a following article, I will demonstrate how to create a small rootfs using [BusyBox](https://busybox.net/).

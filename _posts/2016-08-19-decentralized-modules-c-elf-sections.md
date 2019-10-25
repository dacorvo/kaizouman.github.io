---
layout: post
title: 'Decentralized modules declarations in C using ELF sections'
author: 'David Corvoysier'
date: '2016-08-17 16:00:00'
categories:
- System
tags:
- ELF
- sections
type: post
redirect_from: /2016/08/decentralized-modules-c-elf-sections/
---

In modular programming, a standard practice is to define common interfaces allowing the same type of operation to be performed on a set of otherwise independent modules.

~~~~

modules = [a,b,...]

for each m in modules:
    m.foo
    m.bar

~~~~

To implement this pattern, two mechanisms are required:

- instantiation, to allow each module to define an 'instance' of the common interface,
- registration, to allow each module to 'provide' this instance to other modules.

Instantiation is typically supported natively in high-level languages.

Registration is more difficult and usually requires specific code to be written, or relying on external frameworks. 

Let's see how these two mechanisms can be implemented for C programs.

<!--more-->

>Note: the code snippets in this post can be browsed on the following github [repo](https://github.com/kaizouman/c_modules_section_sample)

## Interface instantiation

In C programs, interface instantiation is implemented using function pointers: basically, the common interface is specified using a struct whose members are the functions that needs to be implemented.

~~~~

module.h:

struct module {
	void (*foo)(void*);
	int (*bar)(char*);
};

a.c:

#include "module.h"

struct module module_a = {
	.foo = foo_a;
	.bar = bar_a;
};

static void foo_a()
{
	
}

static void bar_a()
{

}

b.c:

#include "module.h"

struct module module_b = {
	.foo = foo_b;
	.bar = bar_b;
};

static void foo_b()
{
	
}

static void bar_b()
{

}
~~~~

## Interface registration

The goal here is to allow client code to be able to 'find' the interface instances provided by the modules.

The first question we need to address is whether we register interfaces statically at design time or dynamically at runtime.

Some systems like Linux provide mechanisms for special 'constructors' functions to be called at program initialization. We could take advantage of that feature to allow each module to register its interfaces: see a full example [here](https://github.com/idjelic/lttng2lxt).

In this article, I assume that we are on a system without such capability, and that we as a consequence can only rely on static registration.

>Note that static registration is also more effective, and always desirable on devices with limited hardware.

A first solution for static registration of modules is to give the client code a direct access to the interface instances, by exposing them in public headers.

~~~~

module.h:

struct module {
	void (*foo)();
	void (*bar)();
};

extern struct module module_a;
extern struct module module_b;

foo.c

#include "module.h"

void foo()
{
    module_a.foo();
    module_b.foo();
}

bar.c

#include "module.h"

void bar()
{
    module_a.bar();
    module_b.bar();
}

~~~~

This works, but it is not quite satisfactory: as more modules are added to the program, the client code needs to be modified.

A better solution would be to store the instances anonymously in a static array:

~~~~

module.h:

struct module {
	void (*foo)(void*);
	int (*bar)(char*);
};

extern struct module *modules[];

extern int modules_size;

module.c:

#include "module.h"

extern struct module module_a;
extern struct module module_b;

struct module *modules[2] = {
	&module_a,
	&module_b
};

int modules_size = 2;

foo.c

#include "module.h"

void foo()
{
	int i;
	for (i = 0; i < modules_size; i++) {
		modules[i]->foo();
	}
}

bar.c

#include "module.h"

void bar()
{
	int i;
	for (i = 0; i < modules_size; i++) {
		modules[i]->bar();
	}
}

~~~~

This is quite neat, as we will only need to modify the `module.c` file when a new module is added.

This could be even better though: what if we could add modules without editing any other files ?  

## Taking advantage of ELF sections to create decentralized module tables

The only reason why we need to edit the `modules.c` file is because we need to add new entries to the global modules static array.

The array in itself is just a bunch of pointers written one after the other in a contiguous memory space: what if we could find a way to populate it directly from the modules themselves ?

This cannot be achieved by either the preprocessor or the compiler, as they process compilation units atomically (when a file is processed, the compiler has no knowledge of the other files it has compiled or will compile in the future).

The linker however has the knowledge of all symbols declared in the program, and is even capable of grouping them according to section definitions, as long as we specify them in a custom linker script.

We can take advantage of that to make sure that all references to the interface instances are stored in the same section, and define the modules array as being the start address of the section.

~~~~

module.lds:

SECTIONS
{
	.modules : {
		modules_start = .;
		*(.modules)
		modules_end = .;
	}
}
INSERT AFTER .rodata;

module.h:

struct module {
	void (*foo)();
	void (*bar)();
};

extern const struct module modules_start[];
extern const struct module modules_end[];

Makefile:

OBJS := main.o a.o b.o foo.o bar.o

modules: $(OBJS)
	gcc -o $@ -T module.lds $(OBJS:*.c=*.o)

~~~~

What we do here is that we add an extension to the generic linker script to add a `modules` section.
We also insert two labels at the beginning and end of the section that can be accessed from the C code.

In the `module.h` file, we use these labels to declare external references to the start and end of the section.

> Note that the external references have to be declared as arrays, and not pointers, to make sure the compiler maps correctly the address to the memory region containing the modules: should we have declared them as pointers, the compiler would have mapped the beginning of the memory region to a pointer, then dereferenced it to get access to the modules.
> You can refer to [this post](http://eli.thegreenplace.net/2009/10/21/are-pointers-and-arrays-equivalent-in-c) for a really good explaination of differences between arrays and pointers.

The modules have to be slightly nodified, to make sure they assign their interfaces to the new section:

~~~~

a.c:

...
struct module __attribute__ ((section (".modules"))) module_a = {
	.foo = foo_a,
	.bar = bar_a
};

b.c:

...
struct module __attribute__ ((section (".modules"))) module_b = {
	.foo = foo_a,
	.bar = bar_a
};

~~~~

The syntax is quite ugly, so you probably would hide it inside a preprocessor macro in the `module.h` file.

~~~~
#define DECLARE_MODULE(name, ...) \
    struct module __attribute__ ((section (".modules"))) name = { __VA_ARGS__ };
~~~~

Now we just have to access the global array from the client code using the variables defining its boundaries:

~~~~

foo.c:

#include "module.h"

void foo()
{
	const struct module *m = modules_start;
	while (m < modules_end) {
		m->foo();
		m++;
	}
}

bar.c:

#include "module.h"

void bar()
{
	const struct module *m = modules_start;
	while (m < modules_end) {
		m->bar();
		m++;
	}
}

~~~~

What we have now is a modules framework that can be extended without modifying its core. The modules registraton being static, this is greatly effective both in terms of RAM and CPU consumption.  

## Pitfalls with interface sections

There are a few things that you need to be aware of when using this framework.

First, you need to make sure that the linker aligns the modules in the same way the compiler would: otherwise when going through the table, you may shift and access the wrong data.

This is usually taken care of by enforcing alignment in the linker script:

~~~~

module.lds:

SECTIONS
{
	.modules ALIGN(8) : {
		modules_start = .;
		*(.modules)
		modules_end = .;
	}
}
INSERT AFTER .rodata;
~~~~

Second, depending on your your link configuration, your modules section may be optimized out, as the linker has no way of knowing that it is actually used.

In particular, the `--gc-sections` options will for sure make your table disappear.

The workaround is to explicitly tell the linker that it should keep these symbols:

~~~~

module.lds:

SECTIONS
{
	.modules : {
		modules_start = .;
		*KEEP((.modules))
		modules_end = .;
	}
}
INSERT AFTER .rodata;
~~~~

Last, if some of your modules are distributed as static libraries, the linker may also optimize out the corresponding symbols when linking the whole binary.

The workaround in that case is to prevent optimization by using the linker `--whole-archive` option.


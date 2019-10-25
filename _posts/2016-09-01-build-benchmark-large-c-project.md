---
layout: post
title: 'Benchmarking build systems for a large C project'
author: 'David Corvoysier'
date: '2016-09-01 16:00:00'
categories:
- System
tags:
- build
- make
- ninja
- kbuild
- cmake
type: post
redirect_from: /2016/09/build-benchmark-large-c-project/
---

The performance of build systems has been discussed at large in the developer community, with a strong emphasis made on the limitations of the legacy Make tool when dealing with large/complex projects.

I recently had to develop a build-system to create firmwares for embedded targets from more than 1000 source files.

The requirements were to use build recipes that could be customized for each directory and file in the source tree, similar to what the Linux Kernel does with [kbuild](https://www.kernel.org/doc/Documentation/kbuild/makefiles.txt).

I designed a custom recursive Make solution inspired by [kbuild](https://www.kernel.org/doc/Documentation/kbuild/makefiles.txt).

<!--more-->

>Note: for those interested, this is the build system used in the [Intel Curie SDK for wearables](https://github.com/CurieBSP/main).

After one major release, I had some time to muse around the abundant litterature on build systems, and in particular the infamous ["Recursive Make considered harmful"](http://aegis.sourceforge.net/auug97.pdf), and started to wonder whether I had made the right design choice.

Obviously, my solution had the same limitation that all recursive make have: it was unable to export explicit dependency from one part of the tree to another, but we had easily overcomed that by relying solely on headers to express dependencies , and taking advantage of the [GCC automatic dependencies](http://www.evanjones.ca/makefile-dependencies.html), pretty much like all C projects do anyway.

The solution was also relatively fast, which seemed to contradict the claims of many people](http://stackoverflow.com/questions/559216/what-is-your-experience-with-non-recursive-make).

I therefore decided to do a little benchmark to sort it out.

>You can check for yourself the several solutions in this [repo](https://github.com/kaizouman/build-benchmark).

# The benchmark

The benchmark is to compile a hierachical source tree with directories containing each two source files (header + implementation), and one build fragment specifying a custom preprocessor definition. Each directory implementation 'depends' on its children directories sources by including their headers.

>Yes, it is a wacky design, but I just wanted to challenge the build-system

The benchmark script tests several build-system invocations in four configurations:

- cold start (full build from a fresh tree),
- full rebuild (touch all sources and rebuild),
- build leaf (only touch one of the leaf headers),
- nothing to do.

# The solutions

## Kbuild

The first solution is a variant of my kbuild clone. The design is dead simple:

- each directory has a Makefile fragment that produces a C static library,
- a directory archive aggregates the object files in this directory and the static libraries of its subdirectories,
- a generic Makefile is launched recursively on the source tree to generate libraries and aggregate them to the top.

The syntax of the Makefile fragments is the same as the one used by the Linux Kernel:

~~~~
obj-y = foo.c bar/
cflags-y = -Isomepath -DFOO
~~~~

The generic Makefile is a bit cryptic for those not familiar with the Make syntax, but it actually not very complicated.

This Makefile starts by including the Makefile fragment, then does some processing on the local obj-y variable, to identify local objects and subdirectories.

It then defines rules to:

- build subdirectory archives by relaunching itself on each subdirectory,
- build local objects, taking into account local CFLAGS,
- create the directory library as a 'thin' archive, ie a list of references to actual object files.

~~~~
THIS_FILE := $(abspath $(lastword $(MAKEFILE_LIST)))

all:

# Those are supposed to be passed on the command line
OUT ?= build
SRC ?= src

# Look for a Makefile in the current source directory
-include $(SRC)/Makefile

# First, identify if there are any directories specifed in obj-y that we need
# to descend into
subdir-y := $(sort $(patsubst %/,%,$(filter %/, $(obj-y))))

# Next, update the list of objects, replacing any specified directory by the
# aggregated object that will be produced when descending into it
obj-y := $(patsubst %/, %/built-in.a, $(obj-y))

# Prepend the subdirectories with the actual source directory
subdir-y := $(addprefix $(SRC)/,$(subdir-y))

# Prepend the objects with the actual build DIR
obj-y := $(addprefix $(OUT)/$(SRC)/,$(obj-y))

# Fake target used to force subdirectories to be visited on every Make call
.FORCE:
# Go into each subdirectory to build aggregated objects
$(OUT)/$(SRC)/%/built-in.a: .FORCE
	$(MAKE) -f $(THIS_FILE) SRC=$(SRC)/$* OUT=$(OUT)

# Include dependency files that may have been produced by a previous build
-include $(OUT)/$(SRC)/*.d

# Evaluate local CFLAGS
LOCAL_CFLAGS := -MD $(CFLAGS) $(cflags-y)

# Build C files
$(OUT)/$(SRC)/%.o: $(SRC)/%.c
	mkdir -p $(OUT)/$(SRC)
	$(CC) $(LOCAL_CFLAGS) -c -o $@ $<

# Create an aggregated object for this directory
$(OUT)/$(SRC)/built-in.a: $(obj-y)
	mkdir -p $(OUT)/$(SRC)
	$(AR) -rcT $@ $^

all: $(OUT)/$(SRC)/built-in.a
~~~~

Note that since we cannot 'guess' if a nested library needs to be rebuilt, we force going into each subdirectory using a fake target. This is the main drawback of this solution, as every single directory of the source tree will be parsed even if no file has changed in the source tree.

The top-level Makefile has only two targets:

- one to create the target executable based on the top aggregated library,
- one to create the library by launching the generic Makefile at the top of the source tree.

~~~~
all: $(OUT)/foo

$(OUT)/foo: $(OUT)/built-in.a
        $(CC) -o $@ $^

$(OUT)/built-in.a: .FORCE
        mkdir -p $(OUT)
        $(MAKE) -C $(SRC) -f $(CURDIR)/Makefile.kbuild \
                SRC=. \
                OUT=$(OUT)

.FORCE:
~~~~

## Non recursive Makefile

The second solution is one that is inspired by the principles of Peter Miller's paper.

It uses the same Makefile fragments, but instead of recursively launching Make on subdirectories, it recursively includes the fragments.

The whole process is implemented using a recursive [GNU Make template](https://www.gnu.org/software/make/manual/html_node/Eval-Function.html).

For performance reason, we use a single parameterized generic rule to build objects in the source tree.

During the evaluation of each subdirectory, we gather object files in a global variable, and customize the generic build rule by defining the value
of the CFLAGS for each object in the subdirectory.

>I first designed a variant that created a rule for each subdirectory, but its performances decreased exponentially with the number of directories.

At the end of the Makefile, we use a single `foreach` instruction to include dependency files based on the list of objects.

>I also tried to include these during the subdirectories evaluation, but it was less performant

~~~~
# These are actually passed to us, but provide default values for easier reuse
OUT ?= build
SRC ?= src

# We parse each subdirectory to gather object files
OBJS :=

# Sub-directory parsing function
define parse_subdir

# Reset sub-Makefile variables as a precaution
obj-y :=
cflags-y :=

# Include sub-Makefile
include $(1)/Makefile

# Isolate objects from subdirectories and prefix them with the output directory
_objs := $$(addprefix $(OUT)/$(1)/,$$(sort $$(filter-out %/, $$(obj-y))))

# Define a specific CFLAGS for objects in this subdir
$$(_objs): SUBDIR_CFLAGS := -MD $$(CFLAGS) $$(cflags-y)

# Add subdir objects to global list
OBJS += $$(_objs)

# Isolate subdirs from objects and prefix them with source directory
_subdirs := $$(addprefix $(1)/,$$(sort $$(patsubst %/,%,$$(filter %/, $$(obj-y)))))

# Recursively parse subdirs
$$(foreach subdir,$$(_subdirs), $$(eval $$(call parse_subdir,$$(subdir))))

endef

# Start parsing subdirectories at the root of the source tree
$(eval $(call parse_subdir,$(SRC)))

# Generic rule to compile C files
$(OUT)/%.o: %.c
        mkdir -p $(dir $@)
        $(CC) $(SUBDIR_CFLAGS) -c -o $@ $<

# Include GCC dependency files for each source file
$(foreach obj,$(OBJS),$(eval -include $(obj:%.o=%.d)))

~~~~

The top-level Makefile just includes the "subdirectories" Makefile.

~~~~
all: $(OUT)/foo

include $(CURDIR)/Makefile.subdir

$(OUT)/foo: $(OBJS)
        $(CC) -o $@ $^
~~~~

>It could be a single Makefile, but I found it neater to keep the "generic" template in a separate file.

## Custom generated Makefile

As a variant to the previous solution, I tried to parse the Makefile fragments only once to generate a Makefile in the output directory, then generate the target.

This is basically the same template that is used to generate the actual Makefile: the only difference is that the list of objects and custom CFLAGS per directory are evaluated in memory AND written to the actual Makefile.

~~~~
# These are actually passed to us, but provide default values for easier reuse
OUT ?= build
SRC ?= src

# The only goal of this Makefile is to generate the actual Makefile
all: $(OUT)/Makefile

...


$(OUT)/Makefile::
        @echo "Generating $@"
        @echo "all: $(OUT)/foo" >> $@

# Sub-directory parsing function
define parse_subdir

...

# Include sub-Makefile
include $(1)/Makefile

...

# Define a specific CFLAGS for objects in this subdir
$(1)_CFLAGS := -MD $$(CFLAGS) $$(cflags-y)
# Insert the corresponding goal modifier in the target Makefile
$(OUT)/Makefile::
        echo "$(OUT)/$(1)/%.o: LOCAL_CFLAGS=$$($(1)_CFLAGS)" >> $$@

...

endef

# Start parsing subdirectories at the root of the source tree
$(eval $(call parse_subdir,$(SRC)))

# Finalize target Makefile inserting generic C compilation rule and GCC
# dependencies for each source file
$(OUT)/Makefile::
        echo "OBJS:= $(OBJS)" >> $@
        echo "$(OUT)/%.o: %.c" >> $@
        echo '  mkdir -p $$(dir $$@)' >> $@
        echo '  $$(CC) $$(LOCAL_CFLAGS) -c -o $$@ $$<' >> $@
        echo "" >> $@
        $(foreach obj,$(OBJS),echo "-include $(obj:%.o=%.d)"; >> $@)
        @echo "$(OUT)/foo: $(OBJS)" >> $@
        @echo ' $$(CC) -o $$@ $$^' >> $@
        @echo "Done $@"
~~~~

The top-level Makefile includes the generated Makefile and provides a rule to generate it: GNU Make take cares of the rest.

~~~~
all: $(OUT)/foo

$(OUT)/foo: $(OUT)/Makefile .FORCE
        $(MAKE) -C $(OUT)

FRAGMENTS := \
        $(shell find $(SRC) -name Makefile -cnewer $(OUT)/Makefile 2>/dev/null)

$(OUT)/Makefile: $(FRAGMENTS)
        mkdir -p $(OUT)
        $(MAKE) -C $(SRC) -f $(CURDIR)/Makefile.gen \
                SRC=$(SRC) \
                OUT=$(OUT)

.FORCE:
~~~~

>Note the trick to make sure the Makefile is properly regenerated: since Make has difficulties to cope with a large number of dependencies, we use the shell to identify the fragments that have changed.

## CMake

CMake is a Makefile generator. I added a CMake solution to compare it with the previous custom generated Makefile.

Two issues I had to solve were:

- how to recursively select sources for the final target
- how to express different CFLAGS for a directory

The simple solution I found was to use CMake subdirectories and to define a static library in each one of them.

~~~~
ADD_LIBRARY(output_src_1 STATIC foo.c)
ADD_SUBDIRECTORY(1)
TARGET_LINK_LIBRARIES(output_src_1 output_src_1_1)
...
TARGET_LINK_LIBRARIES(output_src_1 output_src_1_9)
ADD_SUBDIRECTORY(10)
TARGET_LINK_LIBRARIES(output_src_1 output_src_1_10)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D'CURDIR=output/src/1'")
~~~~

>It seems to lead CMake to create a recursive Makefile. It would be interesting to try a different approach using include to gather fragments and per-source properties to set the CFLAGS.

The top-level Makefile has two rules: one to build the generated Makefile, the other one to create the target using it.

~~~~
$(OUT)/foo: $(OUT)/Makefile .FORCE
        $(MAKE) -C $(OUT)

-include $(OUT)/Makefile

$(OUT)/Makefile:
        mkdir -p $(OUT)
        cd $(OUT) && cmake -Wno-dev $(SRC)

.FORCE:
~~~~

>Note that the generated Makefile will detect automatically changes made to the Makefile fragments and regenerate the target Makefile thanks to CMake built-in checks.

## Boilermake

[Boilermake](https://github.com/dmoulding/boilermake) is an awesome generic non-recursive Make template. I included it in order to compare it to my own non-recursive solution.

## Cninja (CMake + Ninja)

CMake is able to generate [Ninja](https://ninja-build.org/) files, so I only had to adapt my CMake based-solution to compare the generated GNU Make build with the generated [Ninja](https://ninja-build.org/) build.

One issue I had with Ninja is that it doesn't cope well with large command lines.

There is an ugly fix that was introduced to address that for the WebKit project.

~~~~
set(CMAKE_NINJA_FORCE_RESPONSE_FILE 1)
~~~~

>Guys, gotcha: when are you gonna fix this ?

## Ninja

The CMake generated Ninja build performance was awesome for incremental builds, but not so good for full builds as soon as the number of files increased.

I had my suspicions it may come from the way CMake generated the Ninja file, especially with the intermediate libraries I had to declare.

Having received a similar feedback from the ninja mailing-list, I wrote a small parser in python to generate the `build.ninja` file directly from the Makefile fragments.

Here is how the generated file looks:

~~~~
rule cc
  deps = gcc
  depfile = $out.d
  command = cc -MD -MF $out.d $cflags -c $in -o $out
rule ld
  command = cc @$out.rsp -o $out
  rspfile = $out.rsp
  rspfile_content = $in
build /home/david/dev/make-benchmark/output/src/main.o: cc /home/david/dev/make-benchmark/output/src/main.c
  cflags = -D'CURDIR=output/src'
build /home/david/dev/make-benchmark/output/src/foo.o: cc /home/david/dev/make-benchmark/output/src/foo.c
  cflags = -D'CURDIR=output/src'
build /home/david/dev/make-benchmark/output/src/1/foo.o: cc /home/david/dev/make-benchmark/output/src/1/foo.c
  cflags = -D'CURDIR=output/src/1'

...

build foo : ld /home/david/dev/make-benchmark/output/src/main.o ...

~~~~

The results are indeed much better, as you will see in the next paragraph.

# The raw results

I ran the benchmark on a Intel Core i7 with 16 GB RAM and an SSD drive.

All build times are in seconds.

~~~~
$make --version
GNU Make 3.81
$cmake --version
cmake version 2.8.12.2
$ninja --version
1.3.4
~~~~

Tree = 2 levels, 10 subdirectories per level (12 .c files)

~~~~
|               | kbuild | nrecur | static | cmake | b/make | cninja | ninja |
|---------------|--------|--------|--------|-------|--------|--------|-------|
| cold start    |  0.08  |  0.06  |  0.08  | 0.55  |  0.08  |  0.36  | 0.08  |
| full rebuild  |  0.06  |  0.06  |  0.06  | 0.23  |  0.07  |  0.04  | 0.06  |
| rebuild leaf  |  0.04  |  0.03  |  0.03  | 0.16  |  0.04  |  0.05  | 0.02  |
| nothing to do |  0.01  |  0.00  |  0.00  | 0.06  |  0.01  |  0.00  | 0.00  |
~~~~

Tree = 3 levels, 10 subdirectories per level (112 .c files)

~~~~
|               | kbuild | nrecur | static | cmake | b/make | cninja | ninja |
|---------------|--------|--------|--------|-------|--------|--------|-------|
| cold start    |  0.47  |  0.45  |  0.51  | 1.84  |  0.52  |  0.91  | 0.53  |
| full rebuild  |  0.48  |  0.46  |  0.44  | 1.34  |  0.54  |  0.39  | 0.32  |
| rebuild leaf  |  0.10  |  0.09  |  0.09  | 0.46  |  0.11  |  0.07  | 0.05  |
| nothing to do |  0.06  |  0.05  |  0.06  | 0.40  |  0.07  |  0.00  | 0.01  |
~~~~

Tree = 4 levels, 10 subdirectories per level (1112 .c files)

~~~~
|               | kbuild | nrecur | static | cmake | b/make | cninja | ninja |
|---------------|--------|--------|--------|-------|--------|--------|-------|
| cold start    |  4.62  |  4.57  |  5.78  | 16.72 |  5.48  |  7.50  |  4.00 |
| full rebuild  |  4.85  |  4.57  |  4.78  | 15.12 |  5.56  |  6.39  |  3.90 |
| rebuild leaf  |  0.98  |  0.86  |  1.04  |  4.47 |  1.07  |  0.28  |  0.21 |
| nothing to do |  0.53  |  0.67  |  0.82  |  4.44 |  0.88  |  0.05  |  0.03 |
~~~~

Tree = 5 levels, 10 subdirectories per level (11112 .c files)

~~~~
|               | kbuild | nrecur | static | cmake  | b/make | cninja | ninja |
|---------------|--------|--------|--------|--------|--------|--------|-------|
| cold start    |  59.01 |  54.07 | 118.00 | 509.96 |  72.41 | 175.58 | 46.98 |
| full rebuild  |  63.41 |  61.38 | 103.95 | 376.40 |  80.17 | 101.76 | 46.66 |
| rebuild leaf  |  10.86 |  17.18 |  59.03 | 215.44 |  20.19 |   2.81 |  2.28 |
| nothing to do |   5.13 |  14.95 |  56.87 | 220.49 |  17.78 |   0.47 |  0.03 |
~~~~

# My two cents

From the results above, I conclude that:

- for my use case, and with my hardware (I suspect SSD is a huge bonus for recursive Make), non-recursive and recursive Makefiles are equivalent,
- my generated Makefile is completely suboptimal (would need to investigate),
- CMake generated Makefiles are pretty darn slow ...
- As long as you don't generate the `build.ninja` with CMake, Ninja is faster than any Make based solution, especially when only a few files have changed.


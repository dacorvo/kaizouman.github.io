---
layout: post
title: 'Fixed-point representation for quantization'
author: 'David Corvoysier'
date: '2023-05-26 12:00:00'
categories:
- Machine Learning
tags:
- quantization
type: post
redirect_from: /2023/05/quantization-fixed-point/
---

As explained in my introduction to [Machine Learning quantization](/2023/05/ml-quantization-introduction.html#quantized-linear-operations),
 the quantization of a ML model produces a graph of operations applied on quantized tensors.

Quantized tensors are actually integer tensors that share the same float scale and integer zero-point.

The implementation of the quantized operations is device-specific.

One of the main design decision is how the inputs, weights and output float scales are propagated and applied in the quantized graph.

In two other posts I will explain how is is possible to use integer arithmetic operators for that purpose if the scales are represented
as fixed-point numbers.

This posts is a brief introduction to the fixed-point representation and to the fixed-point arithmetic operators.

<!--more-->

## Fixed-point representation

Before the introduction of the floating point representation, decimal values were expressed using a fixed-point representation.

This representation also uses a mantissa and an exponent, but the latter is implicit: it defines the number of bits in the mantissa
dedicated to the fractional part of the number.

The minimum non-zero value that can be represented for a given number of fractional bits is $2^{-fracbits}$.

For instance, with three fractional bits, the smallest float number than can be represented is $2^{-3} = 0.125$.

Below is an example of an unsigned 8-bit fixed-point number with 4 fractional bits.

<pre class='diagram'>
.------------------------------------.  
|  0   1   0   1 |  1   1   1    0   |
.------------------------------------.  
|  integer bits  |  fractional bits  |
.------------------------------------.  
|  3   2   1   0 | -1  -2  -3   -4   |
'------------------------------------'  
</pre>

The decimal value of that number is: $2^{2} + 2^{0} + 2^{-1} + 2^{-2} + 2^{-3} = 5.875$

The precision of the representation is directly related to the number of fractional bits.

Below are some more examples of PI represented with unsigned 8-bit fixed-point numbers different fractional bits:

| float    | frac_bits | mantissa |  binary  |
|----------|-----------|----------|----------|
| 3.140625 | 6         | 201      | 11001001 |
| 3.15625  | 5         | 101      | 01100101 |
| 3.125    | 4         | 50       | 00110010 |
| 3.125    | 3         | 25       | 00011001 |
| 3.25     | 2         | 13       | 00001100 |
| 3.0      | 1         | 6        | 00000110 |


## Obtaining a fixed-point representation of a float

As a reminder, a float number is represented as:

$$x = mantissa * 2^{exponent}$$

Our goal here is to obtain a fixed-point representation of $x$.

Technically, we could directly take the float mantissa, but it is 24-bit, with a high risk of overflows in the downstream
fixed-point operations.

For the range of numbers used in Machine Learning, an 8-bit mantissa is usually enough to accurately represent a `float32` number.

As a consequnce, we only need to keep the 8 most significant bits of the mantissa, which effectively means quantizing the float to
8-bit with the power-of-two scale that minimizes the precision loss.

This can be achieved in several ways depending on the level of abstraction you are comfortable with: below is an algorithm
relying only on high-level mathematical operations.

```python


def to_fixed_point(x, bitwidth, signed=True):
    """Convert a number to a FixedPoint representation

    The representation is composed of a mantissa and an implicit exponent expressed as
    a number of fractional bits, so that:

    x ~= mantissa . 2 ** -frac_bits

    The mantissa is an integer whose bitwidth and signedness are specified as parameters.

    Args:
        x: the source number or array


    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    # Evaluate the number of bits available for the mantissa
    mantissa_bits = bitwidth - 1 if signed else bitwidth
    # Evaluate the number of bits required to represent the whole part of x
    # as the power of two enclosing the absolute value of x
    # Note that it can be negative if x < 0.5
    whole_bits = np.ceil(np.log2(np.abs(x))).astype(np.int32)
    # Deduce the number of bits required for the fractional part of x
    # Note that it can be negative if the whole part exceeds the mantissa
    frac_bits = mantissa_bits - whole_bits
    # Evaluate the 'scale', which is the smallest value that can be represented (as 1)
    scale = 2. ** -frac_bits
    # Evaluate the minimum and maximum values for the mantissa
    mantissa_min = -2 ** mantissa_bits if signed else 0
    mantissa_max = 2 ** mantissa_bits - 1
    # Evaluate the mantissa by quantizing x with the scale, clipping to the min and max
    mantissa = np.clip(np.round(x / scale), mantissa_min, mantissa_max).astype(np.int32)
    return mantissa, frac_bits


```

The algorithm above produces a fixed-point representation of $x$ such that:

$$x_{float} \approx x_{int} . 2^{-x_{fracbits}}$$


## Fixed-point addition (or subtraction)

The reason why the fixed-point representation comes to mind when it comes to quantization is that it has exactly the same
restrictions regarding the addition of numbers: they must be expressed using the same amount of fractional bits.

The addition can then be performed directly on the underlying integer.

The resulting sum is a fixed-point number with the same fractional bits. It is exact unless it overflows.

What is really interesting here is that the alignment of fixed-point numbers is trivial: it can just be performed
using a left bitshift.

Example:

The following fixed-point (values, fractional bits) pairs represent the following float values:

$a: (84, 3) = 84 * 2^{-3} = 10.5​$

$b: (113, 4) = 113 * 2^{-4} = 7.0625$

​
Before summing a and b, we need to shift $a$ to the left to align it with $b$:

$s = a + b = 84 << 1 + 113 = 168 + 113 = 281$

The sum is a fixed-point number with 4 fractional bits:

$s: (281, 4) = 281 * 2^{-4} = 17.5625​$


## Fixed-point multiplication

The multiplication of two fixed-point numbers can be performed directly on the underlying integer numbers.

The resulting product is a fixed-point number with a number of fractional bits corresponding to the sum of the fractional bits of the inputs. It is exact unless it overflows.

Example:

Going back to our two numbers:

$a: (84, 3) = 84 * 2^{-3} = 10.5​$

$b: (113, 4) = 113 * 2^{-4} = 7.0625$

Their fixed-point product is:

$p = a.b = (84 . 113, 3 + 4) = (9492, 7) = 74.15625$


## Fixed-point downscale

The mantissa of the resulting product of two fixed-point numbers can go very quickly, which would eventually lead to an overflow when chaining multiple operations.

It is therefore common to 'downscale' the result of a multiplication using a right-shift.

Example:

Going back to our previous product:

$p = a.b = (84 . 113, 3 + 4) = (9492, 7) = 74.15625$

It can be downscaled to fit in 8-bit by shifting right and adjusting the fractional bits:

$downscale(p) = p >> 6 = (148, 1) = 74$

Note that the right-shift operation always perform a floor, which may lead to a loss of precision.

For that reason, it is often implemented as a 'rounded' right-shift by adding $2^{n-1}$ before shifting of $n$.

>Note: this is mathematically equivalent to adding $0.5$ to $\frac${x}{2^{n}}$ before taking its floor.


## Fixed-point division

The division of two fixed-point numbers can be performed directly on the underlying integer numbers.

The resulting quotient is a fixed-point number with a number of fractional bits corresponding to the subtraction of the fractional bits of the inputs. It is usually not exact.

Example:

Going back to our two numbers:

$a: (84, 3) = 84 * 2^{-3} = 10.5​$

$b: (113, 4) = 113 * 2^{-4} = 7.0625$

Their fixed-point division is:

$p = \frac{b}{a} = (\frac{113}{84}, 4 - 3) = (1, 1) = 0.5$

A possible mitigation is to left-shift the dividend before the division to increase its precision: the resulting quotient will in turn have an increased precision.

$b: (113, 4) << 3 = (113 << 3, 4 + 3) = (904, 7) = 904 * 2^{-7} = 7.0625$

$p = \frac{b}{a} = (\frac{904}{84}, 7 - 3) = (10, 4) = 0.625$

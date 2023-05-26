---
layout: post
title: 'Aligning quantization scales before incompatible operations'
author: 'David Corvoysier'
date: '2023-05-26 12:00:00'
categories:
- Machine Learning
tags:
- quantization
type: post
redirect_from: /2023/05/align-quantization-scales/
---

As explained in my introduction to [Machine Learning quantization](/2023/05/ml-quantization-introduction#quantized-linear-operations),
 important restrictions apply to operations performed on quantized inputs.

First, additions between the integer mantissa of quantized inputs can only be performed if they are in the same scale.

This comes from the representation of the quantized numbers:

$a = (n - zeropoint_a) * scale_a$

$b = (m - zeropoint) * scale_b$

$a$ and $b$ integer mantissa can only be added if $scale_a == scale_b$, allowing us to write directly:

$a + b = (n - zeropoint_a + m - zeropoint_b) * scale_a$

Intuitively, this is equivalent to say that you cannot add bytes to kilobytes without converting one number representation to the other.

<!--more-->

Second, the same kind of restriction can be extended to operations that combine the channels of the inputs, such as the Matrix Multiplication or the
Convolution.

For such operations, the channels must be all in the same scale: in other words, the inputs of these operations mut be quantized per-tensor.

The first restriction is a major issue for all Machine Learning models but the sequential ones. In other words it is a major issue for all models
of the 2020's, as they all include parallel branches that are eventually merged with an addition layer.

The second restriction used to be rather harmless: most models used to have very homogeneous activation, most of the time allowing a lossless
quantization to 8-bit per-tensor.

This changed with the introduction of Transformer models, whose activation ranges can vary with a factor from 1 to 100 between channels, making
per-tensor quantization less efficient.

In the next paragraphs I will detail a method to solve the first issue using only integer operations.

## Fixed-point representation

Before the introduction of the floating point representation, decimal values were expressed using a fixed-point representation.

In a nuthshell, a fixed-point representation is composed of a mantissa and an implicit exponent.

The implicit exponent defines the number of bits dedicated to the fractional part of the number in the mantissa.

The minimum non-zero value that can be represented for a given number of fractional bits is $2^{-fracbits}$.

For instance, with three fractional bits, the smallest float number than can be represented is $2^{-3} = 0.125$.

Below are some examples of numbers represented with different fractional bits:

| float | frac_bits | integer | binary |
|-------|-----------|---------|--------|
| 3.625 | 3         | 29      | 11101  |
| 3.5   | 2         | 14      | 1110   |
| 3.5   | 1         | 7       | 111    |

The reason why the fixed-point representation comes to mind when it comes to quantization is that it has exactly the same
restrictions regarding the addition of numbers: they must have the exact same number of fractional bits.

The reason why it is really interesting here is because the alignment of fixed-point numbers is trivial: it can just be performed
using a bitshift.

Example:

The following fixed-point (values, fractional bits) pairs represent the following float values:

$a: fixed-point(84, 3) = 84 * 2^{-3} = 10.5​$

$b: fixed-point(113, 4) = 113 * 2^{-4} = 7.0625$

​
Before summing a and b, we need to shift $a$ to the left to align it with $b$:

$s = a + b = 84 << 1 + 113 = 168 + 113 = 281$

The sum is a fixed-point number with 4 fractional bits:

$s: fixed-point(281, 4) = 281 * 2^{-4} = 17.5625​$

>Note: if you remove the bits corresponding to the exponent in a float number you obtain a fixed-point with a number of fractional
bits corresponding to the exponent.

## Explicitly apply input scale using fixed-point arithmetics

Going back to our problem, we see immediately that if the scales of the inputs were power-of-two's, then the inputs
could be interpreted as fixed-point numbers, and it would become trivial to align them.

Here comes the trick: it is actually not that difficult to obtain a fixed-point representation of the inputs, even
with a scale that is not a power-of-two.

As a reminder, a quantized number is represented as:

$x = (n - zeropoint) * scale$

Our goal here is to obtain a fixed-point representation of $x$.

The thing is: fixed-point arithmetic operations produce fixed-point numbers, and the first term is already an 8-bit integer,
i.e. a fixed-point with zero fractional bits, so all we have to do is to make sure the scale is a fixed-point number.

Technically, we could directly take the float mantissa, but it is 24-bit, with a high risk of overflows in the downstream
fixed-point operations.

Since the inputs are quantized to 8-bit anyway, an 8-bit mantissa is enough to accurately represent a `float32` scale, so
we only need to keep the 8-bit most significant bits of the mantissa, which effectively means quantizing the float to 8-bit
with the power-of-two scale that minimizes the precision loss.

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

Now that we have a fixed-point representation of the scale as:

$scale \approx i_s . 2^{-fracbits_s}$

We can derive an approximated fixed-point representation of $x$:

$x \approx ((n - zeropoint) * i_s). 2^{-fracbits_s}$

## Align inputs explicitly after converting them to fixed-point

Using the fixed-point scales obtained as specified in the previous paragraph, it is now possible to align
inputs expressed with different scales.

$a \approx ((n - zeropoint_a) * p). 2^{-fracbits_a} = a_i . 2^{-fracbits_a}$

$b \approx ((m - zeropoint_b) * q). 2^{-fracbits_a} = b_i . 2^{-fracbits_b}$


At quantization time, we can evaluate channel-wise the maximum number of fractional bits for the two inputs we
want to combine and produce two relative shifts to be applied to each one of them:

$maxfracbits = max(fracbits_a, fracbits_b)$

$shift_a = fracbits_a - maxfracbits$

$shift_b = fracbits_b - maxfracbits$

Then the sequence of operations before the addition is to:

- convert inputs integer mantissa to a fixed-point representation:

$a_i = (n - zeropoint_a) * p$

$b_i = (m - zeropoint_b) * q$

- align the resulting fixed-point:

$a_i = a_i << shift_a$

$b_i = b_i << shift_b$

- perform the integer addition

$s_i = a_i + b_i$

This produces a fixed-point tensor with an implicit scale of $2^{-maxfracbits}$.

This additional scale needs to be taken into account when quantizing the outputs of the addition.

Mathematically, this means that the scale of the outputs obtained after calibration must be multiplied
by $2^{-maxfracbits}$.

>Note: In another post, I will explain how this can be further simplified if the output quantization scale is
 also represented as a fixed-point number.

## Generalization to per-axis inputs

The same kind of alignment can be applied to inputs quantized per-axis when reaching an operation that requires
per-tensor inputs.

Ths only difference is that the maximum number of fractional bits is a scalar value corresponding to the aligned
per-tensor scale:

$maxfracbits = max(fracbits_a)$
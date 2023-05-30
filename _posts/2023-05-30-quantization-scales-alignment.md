---
layout: post
title: 'Aligning quantization scales before incompatible operations'
author: 'David Corvoysier'
date: '2023-05-30 12:00:00'
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

Intuitively, this is analog to say that you cannot add two quantities expressed in different units (like bytes and kilobytes) without converting one
number representation to the other.

<!--more-->

The same kind of restriction can also be extended to operations that combine the channels of the inputs, such as the Matrix Multiplication or the
Convolution.

For such operations, the channels must be all in the same scale: in other words, the inputs of these operations must be quantized per-tensor.

The first restriction is a major issue for all Machine Learning models that are not purely sequential. In other words it is a major issue for all models
of the 2020's, as they all include parallel branches that are eventually merged with an addition layer.

The second restriction used to be rather harmless: most models used to have very homogeneous activations, allowing a lossless quantization to 8-bit per-tensor.

This changed with the introduction of Transformer models, whose activation ranges can vary with a factor from 1 to 100 between channels, making
per-tensor quantization less efficient.

On devices that support float arithmetics, not being able to use directly the integer mantissa is hardly a problem, except maybe for efficiency.

On devices supporting only integer arithmetics this is a serious issue.

In the next paragraphs I will detail a method to align inputs using only integer operations.

## Explicitly apply input scale using fixed-point arithmetics

In a previous post, I introduced the [fixed-point representation](/2023/05/quantization-fixed-point) and explained how it relates to quantization.

Going back to our problem, we see immediately that if the scales of the inputs were power-of-two's, then the inputs
could be interpreted as fixed-point numbers, and it would become trivial to align them.

Here comes the trick: it is actually not that difficult to obtain a fixed-point representation of the inputs, even
with a scale that is not a power-of-two.

As a reminder, a quantized number is represented as:

$x = (n - zeropoint) * scale$

Our goal here is to obtain a fixed-point representation of $x$.

The thing is: fixed-point arithmetic operations produce fixed-point numbers, and the first term is already an 8-bit integer,
i.e. a fixed-point with zero fractional bits, so all we have to do is to make sure the scale is a fixed-point number.

Since the inputs are quantized to 8-bit anyway, an 8-bit mantissa is enough to accurately represent a `float32` scale, so
we only need to keep the 8-bit most significant bits of the scale mantissa.

You can refer to this [fixed-point conversion algorithm](/2023/05/quantization-fixed-point) for an example of how we can
convert the scale to a fixed-point representation.

Now that we have a fixed-point representation of the scale as:

$scale \approx i_s . 2^{-fracbits_s}$

We can derive an approximated fixed-point representation of $x$:

$x \approx ((n - zeropoint) * i_s). 2^{-fracbits_s}$

Due to the multiplication of the two integers, this representation has a higher bitwidth than the original quantized
number, but it should not be an issue since the resulting mantissa needs to be calculated only when the operation is
 performed, and thus using an intermediate buffer with a larger bitwidth.
 
>Note: If that is an issue, then it could still be reduced using a right bitshift whose magnitude would be evaluated using the
calibration information.

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

>Note: as mentioned in a previous note, I will explain in another post how this can be achieved using integer arithmetics
only.

## Generalization to per-axis inputs

The same kind of alignment can be applied to inputs quantized per-axis when reaching an operation that requires
per-tensor inputs.

Ths only difference is that the maximum number of fractional bits is a scalar value corresponding to the aligned
per-tensor scale:

$maxfracbits = max(fracbits_a)$
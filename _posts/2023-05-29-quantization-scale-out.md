---
layout: post
title: 'Resolve quantization scales after an operation'
author: 'David Corvoysier'
date: '2023-05-29 12:00:00'
categories:
- Machine Learning
tags:
- quantization
type: post
redirect_from: /2023/05/quantization-scale-out/
---

As explained in my introduction to [Machine Learning quantization](/2023/05/ml-quantization-introduction#quantized-linear-operations),
 the inputs, weights and outputs of a quantized operation are quantized each with a different scale.

In the same post, I explain how these scales can be folded into a single output scale, allowing the operation to be performed on the integer mantissa
of the quantized inputs and weights:

$scale_{folded} = \frac{scale_{out}}{scale_{in} . scale_{w}}$

In [another post](/2023/05/quantization-scales-alignment) I explain how heterogenous input scales could be converted to a fixed-point representation
and aligned before the operation, resulting in yet another implicit scale expressed as a power-of-two that needs to be applied to the output scale.

In this post I explain how these output scales can be applied using integer arithmetics only. 

<!--more-->

## Reminder: how are output scales applied in a quantized graph

As a general principle, the last step of a quantized operation is a downscale to reduce the output bitwidth.

When applied to float outputs, the general formula for the downscale is:

$outputs_{uint8} = saturate(round(\frac{outputs_{float32)}}{scale_{out}}) + {zp_{out}})$

For a quantized output of scale $y_{s}$ and zero-point $y_{zp}$.

As explained in my [quantization introduction](/2023/05/ml-quantization-introduction#quantized-linear-operations),
some compatible operations can be applied directly on the integer mantissa of the quantized inputs and weights,
folding the inputs and weights scale into the output scale.

The downscale operation becomes then:


$outputs_{uint8} = saturate(round(\frac{outputs_{int32}}{scale_{folded}}) + zp_{out})$

with $scale_{folded} = \frac{scale_{out}}{scale_{in} . scale_{w}}$

This operation still requires a division and a round that is not easily implemented using integer arithmetic operators.

## Use fixed-point folded scale reciprocal to obtain rescaled fixed-point outputs

The idea is to convert the scale to a fixed-point representation to be able to take advantage of integer arithmetic operators
and obtain a fixed-point representation of the downscaled outputs.

Since the fixed-point division is a lossy operation, instead of dividing by the folded output scale, we can multiply by its reciprocal $\frac{1}{scale_{folded}}$.

The first step is to obtain a fixed-point representation of the reciprocal of the folded scale:

$rec_{folded} = to_fixed_point(\frac{scale_{in}.scale_{w}}{scale_{out}}) = rec_{int} . 2^{-fracbits_{rec}}$

You can refer to this [fixed-point conversion algorithm](/2023/05/quantization-fixed-point) for an example of how we can
convert the scale to a fixed-point representation.

Then the rescaled outputs are simply evaluated as:

$outputs_{int32} = outputs_{int32}.rec_{folded}$

## Reduce the precision of the fixed-point rescaled outputs using a rounded right-shift

The rescaled outputs are represented as a fixed-point number with an implicit scale of $2^{-fracbits_{rec}}$.

To obtain the actual 8-bit integer values corresponding to the original downscale operation, we must apply this implicit
scale.

We use the rounded right-shift operation described in the [fixed-point introduction post](/2023/05/quantization-fixed-point)

$outputs_{int8} = outputs_{int32} + 2^{fracbits_{rec} - 1}>> frac_bits_{rec}$

Then we can apply the zero-point:

$outputs_{uint8} = saturate(outputs_{int8} + zp_{out})$
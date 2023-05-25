---
layout: post
title: 'A brief introduction to Machine Learning models quantization'
author: 'David Corvoysier'
date: '2023-05-25 12:00:00'
categories:
- Machine Learning
tags:
- quantization
type: post
redirect_from: /2023/05/ml-quantization-introduction/
---

## Benefits of quantization for Machine-Learning models

Even before the development of Large Language Models (LLM), the increasing 
memory and computing requirements of Deep Neural Networks (DNN) has been a concern.

Functionally, DNN are graphs of tractable mathematical operations: the inputs are 
fed at the stem and the chain of operations produces the outputs at the head.

From an implementation perspective, the operations are performed on floating point 
numbers, which are a digital representation of "real" numbers composed of a mantissa and an 
exponent:

$x = mantissa . 2^{exponent}$

The 32-bit floating point representation if the most common, as it allows to represent 
numbers in a range that is sufficient for most operations. The `float32` mantissa is composed of 
24-bit (including sign), and the exponent is 8-bit.

Each operation performed at an operating node in the inference device requires its inputs
to be transferred from either a static memory location or the previous processing nodes.

The cost of these transfers adds-up with the cost of the operations themselves. 

The DNN terminology for operation data is "weights" for static inputs and "activations" for dynamic inputs/outputs.

Note: the outputs of an operation are designated as "activations" even if it is not actually an activation.

The process of representating the n-bit weights and activations of a DNN into a smaller 
number of bits is called quantization. It is typically used in DNN to "quantize" `float32` into 8-bit integer. 

This brings several benefits:
- reducing the weights to 8-bit requires 4 times less memory on the device to store them,
- performing operations using 8-bit inputs is often faster (all integer operations except division are faster on CPU and GPU can take advantage of faster 8-bit Tensor cores),
- reducing the activations to 8-bits reduces the amount of data exchanged between nodes, which impacts latency.


## A mathematical formulation of linear quantization

The most widespread type of quantization is the linear quantization.

The representation of a linearly quantized number is composed of:
- an integer mantissa,
- a float scale,
- an integer zero-point.

$x = (mantissa - zeropoint).scale$

The scale is used to project back the integer numbers into a float representation.
The zero point corresponds to the value that zero takes in the target representation.

If we compare that formula with the floating point representation one can see 
immediately that each floating point number can be represented exactly with the same 
mantissa, a scale corresponding to the exponent and a null zero-point.

Of course this representation would be very inefficient because it would require two 
integer and a float to represent each float.

## Applicability of quantization to Machine-Learning

When quantizing Machine-Learning models, one can take advantage of the fact that 
the training produces weights and activations that stay within reasonably stable ranges
for a given operation.

This comes partly from the initialization, partly from the constraints applied 
on the weights (like regularization), and partly because of the explicit normalization 
operations inserted in the graph.

This means that the weights and activations of a single operation can be represented 
using the same scale and zero-point, thus leading to a very compact representation.

One typically distinguish two quantization schemes:
- per-tensor quantisation uses a single scalar value for scale and zero-point for a whole 
tensor of weights or activations,
- per-axis quantization uses a vector of scales and zero-points whose length corresponds 
to the number of channels in the tensor (whatever its spatial dimensions are).

## Quantizing a float tensor

The first step to quantize a float tensor is to choose the quantisation range, i.e. the 
minimum and maximum values one wants to represents.
There are quite a lot of options to evaluate the quantization range, that will not be 
detailed here: let’s just say that the choice should be made by trial and error depending on the 
model.

Note: static weights are typically quantized using their full-range whereas we use an histogram-based
algorithm for dynamic activations.

For a target bit width of n for the mantissa, one evaluates the scale as:

$scale = \frac{Max - Min}{2^n - 1}$

The zero-point is then deduced from the scale to make sure that `Min` is mapped to the 
lowest integer value and `Max` to the highest integer value.

This leads to the following formulas for signed/unsigned representations:

- unsigned: $zeropoint = round(\frac{Min}{scale})$
- signed: $zeropoint = round(\frac{Min}{scale}) - 2^{n - 1}$


The quantization of a float tensor is then:

$mantissa = saturate(round(\frac{x}{scale}) + zeropoint)$

Again, the saturation depends of the signed of the target representation: $[0, 2n - 1]$ for 
an unsigned representation and $[-2^{n-1}, 2^{n-1} - 1]$ for a signed representation.

Note that the zero-point always has the same signedness as the mantissa.

## Quantizing a Machine Learning Model

As mentioned before, a Machine Learning model uses two types of tensors: weights and activations.

The static weights need to be quantized only once, each weight tensor producing three new static
tensors for the mantissa, scale and zeropoint.

Since weights can contain positive and negative values, they are typically quantized into `int8`.

W(`float32`) -> I(`int8`), scale(`float32`), zeropoint(`uint8`)

The dynamic activations on the other hand need to be quantized on-the-fly by inserting the quantization
operations in the graph:

- evaluate the quantization range,
- quantize.

The evaluation of the quantization range is costly because is requires a full-scan of the activations tensor,
which prevents parallelism. For that reason, the activations quantization ranges are often evaluated before
the inference on a selected number of samples: this is called the calibration of the quantized model.

As a result, each activation float variable is mapped to an integer variable and two static tensors.

A(`float32`) -> I(`int8`/`uint8`), scale(`float32`), zeropoint(`uint8`)

Note: the activations can be quantized into either `int8` or `uint8`. It is simpler to quantize them to `uint8`
if they correspond to the output of a ReLU operation, since zero-point will be in that case 0.

Conceptually, the resulting graph is a clone of the original graph where all compatible operations are:
- replaced by a version that operates on tuples of (inputs, scale, zero-point),
- surrounded by quantization operations.

Most basic Machine Learning operations can be performed using integer arithmetics, with the following rules and restrictions:
- zeropoint must be applied at some point in the operation,
- additions between inputs can only be performed if they are in the same scale,
- operations that combine inputs from different input channels can only be performed if the channels are in the same scale, i.e
if the inputs are quantized per-tensor.

## Wrap-up example: a matrix multiplication between quantized inputs and weights

Let's consider a simple matrix multiplication of an $X(I, J)$ input by a $W(J, K)$ set of weights:

$Y = X.W + B$

Since the matrix multiplication multiplies all inputs along the dimension of length $J$ and adds them,
 $X$ cannot be quantized per-axis, because it will lead to the addition of quantized numbers that are not in the same scale. 
There is no such restriction on $W$, since the filters along $K$ are all applied independently.
After quantization of the weights per-axis and calibration of the inputs per-tensor, we obtain:

$X \approx X_s * (X_q - X_{zp})$, with $X_s()$, $X_q(I, J)$, $X_{zp}()$

$W \approx W_s * (W_q - W_{zp})$, with $W_s(K)$, $W_q(J, K)$, $W_{zp}(K)$

$Y \approx Y_s * (Y_q - Y_{zp})$, with $Y_s(K)$, $Y_q(I, K)$, $Y_{zp}(K)$

And the quantized graph equivalent to the float operation would be:

$O = X_s * (X_q - X_{zp}) . W_s * (W_q - W_{zp})$

$Y_q = saturate(round(\frac{O}{Y_s}) + Y_{zp})$

$Y \approx Y_s * (Yq - Y_{zp})$

Since $X_s$ is a scalar, and $W_s$ has the same dimension as the output last dimension,
this can also be written:

$O = (X_s * W_s) * (X_q - X_{zp}) . (W_q - W_{zp})$

$Y_q = saturate(round(\frac{O}{Y_s}) + Y_{zp})$

$Y \approx Y_s * (Yq - Y_{zp})$

This means that the matrix multiplication can be operated equivalently on integer values,
and the result is a quantized integer number with a scale corresponding to the product of
the inputs and weights scale and a null zero-point.

$O_q = (X_q - X_{zp}) . (W_q - W_{zp})$

$O = (X_s * W_s) * O_q$

$Y_q = saturate(round(\frac{O}{Y_s}) + Y_{zp})$

$Y \approx Y_s * (Yq - Y_{zp})$

The question that should immediately arise at this stage is why we need another quantization
operation after the matrix multiplication, since we already have a quantized output ?

The reason is simply the bitwidth of the outputs: we need an explicit quantization to make
sure that the results of the integer matrix multiplication fit in 8-bit.

>Note: when the operation is followed by a bias addition, the biases are typically quantized to
32-bit with a scale precisely equal to $X_s * W_s$ so that they can be added directly to the outputs
before quantizing.

Going one step further and replacing $O$, since $Y_s$ has the same shape as $X_s * W_s$, we can write:

$O_q = (X_q - X_{zp}) . (W_q - W_{zp})$

$Y_q = saturate(round(\frac{X_s * W_s}{Y_s} * O_q) + Y_{zp})$

$Y \approx Y_s * (Yq - Y_{zp})$

This reveals that we can directly quantize the integer outputs of the operation with a scale equal
 to $\frac{Y_s}{X_s * W_s}$.

>Note: Depending on the capabilities of the device, this chain of operations can be implemented in very
different ways: some devices might not have efficient implementations of the integer Matrix Multiplication.
In that case, they could apply immediately the inputs and weights scales to fallback to a float
matrix multiplication. 

In a upcoming article I will explain how it is possible to add two inputs quantized with different scales
by adding an explicit alignment operation beforehand.
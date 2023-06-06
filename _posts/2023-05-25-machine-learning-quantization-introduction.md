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

Even before the development of Large Language Models (LLM), the increasing
memory and computing requirements of Deep Neural Networks (DNN) has been a concern.

Functionally, DNN are graphs of arithmetic operations: the inputs are fed at the
stem and the chain of operations produces the outputs at the head.

From an implementation perspective, the operations are performed on floating point
numbers, which are a digital representation of decimal numbers composed of a mantissa and an
exponent:

$$x = mantissa . 2^{exponent}$$

<!--more-->

The 32-bit floating point representation if the most common, as it allows to represent
numbers in a range that is sufficient for most operations. The `float32` mantissa is composed of
24-bit (including sign), and the exponent is 8-bit.

Each operation performed at an operating node in the inference device requires its inputs
to be transferred from either a static memory location or the previous processing nodes.

The cost of these transfers adds-up with the cost of the operations themselves.

The DNN terminology for operation data is "weights" for static inputs and "activations" for dynamic inputs/outputs.

Note: the outputs of an operation are designated as "activations" even if it is not actually an activation.

The process of representating the n-bit weights and activations of a DNN into a smaller
number of bits is called quantization[^quant].

It is typically used in DNN to "quantize" `float32` weights and activations into 8-bit integer.

This brings several benefits:
- reducing the weights to 8-bit requires 4 times less memory on the device to store them,
- reducing the activations to 8-bits reduces the amount of data exchanged between nodes, which impacts latency,
- using 8-bit instead of 32-bit inputs for an operation improves vectorization (multiple data processed at the same time for a single operation),
- all standard integer arithmetic operations but the division are faster than their floating point counterpart,
- GPU devices may include specific mechanisms to process 8-bit inputs (like NVIDIAS 8-bit Tensor cores).

## A mathematical formulation of linear quantization

The most widespread type of quantization is the *linear* or *affine* quantization scheme first introduced in tensorflow lite[^qtf].

The representation of a linearly quantized number is composed of:
- an integer mantissa,
- a float scale,
- an integer zero-point.

$$x = (mantissa - zeropoint).scale$$

The scale is used to project back the integer numbers into a float representation.

The zero point corresponds to the value that zero takes in the target representation.

If we compare that formula with the floating point representation one can see
immediately that each floating point number can be represented exactly with the same
mantissa, a scale corresponding to the exponent and a null zero-point.

Of course this representation would be very inefficient because it would require two
integer and a float to represent each number.

## Applicability of quantization to Machine-Learning

When quantizing Machine-Learning models, one can take advantage of the fact that
the training produces weights and activations that stay within reasonably stable ranges
for a given operation.

This comes partly from the initialization, partly from the constraints applied
on the weights (like regularization), and partly because of the explicit normalization
operations inserted in the graph (like the BatchNorm).

This means that the weights and activations tensors for a specific operation can be represented
using the same scale and zero-point, thus leading to a very compact representation.

There are various subtypes of quantization.

The first two subtypes are related to the dimensions of the scale and zero-point:
- *per-tensor* quantization uses a single scalar value for scale and zero-point for a whole
tensor of weights or activations,
- *per-axis* quantization uses a vector of scales and zero-points whose length corresponds
to a single axis of the tensor (typically the *channels* or *embeddings* axis).

The second subtypes are related to the *symmetry* of the resulting quantized numbers:
- *symmetric* quantization assumes that the quantization range is symmetric, which leads to a zero-point equal
to zero and a signed integer representation of the values,
- *asymmetric* quantization does not assume anything, and zero-point is typically non-null.

Weights are typically quantized symmetrically per-axis.

Activations are typically quantized asymmetrically, most of the time per-tensor.

## Quantizing a float tensor

The first step to quantize a float tensor is to choose the quantization range, i.e. the
minimum and maximum float values one wants to represent: $[Min, Max]$.

Since the weights are constant tensors, they are typically quantized using the mimimum and maximum
values of the tensor, globally or along the channel axis.

Evaluating the quantization range of the activations is more difficult as they are dependent of the inputs
of the previous operation. Their range is therefore evaluated globally inside a model, as explained in the next
paragraph.

For a target bit width of n for the mantissa, one evaluates the scale as:

$$scale = \frac{Max - Min}{2^n - 1}$$

The zero-point is then deduced from the scale to make sure that $Min$ is mapped to the
lowest integer value and $Max$ to the highest integer value.

This leads to the following formulas for signed/unsigned representations:

- unsigned: $zeropoint = round(\frac{Min}{scale})$
- signed: $zeropoint = round(\frac{Min}{scale}) - 2^{n - 1}$

The quantization of a float tensor is then:

$$mantissa = saturate(round(\frac{x}{scale}) + zeropoint)$$

Again, the saturation depends of the signed of the target representation:
- unsigned: $[0, 2n - 1]$,
- signed: $[-2^{n-1}, 2^{n-1} - 1]$.

Note that the zero-point always has the same signedness as the mantissa.

## Quantizing a Machine Learning Model

As mentioned before, a Machine Learning model uses two types of tensors: weights and activations.

The static weights need to be quantized only once, each weight tensor producing three new static
tensors for the mantissa, scale and zeropoint.

Since weights can contain positive and negative values, they are typically quantized into `int8`.

<pre class='diagram'>
             .----------.
             |  Weights |
             |  float32 |
             | constant |
             +----+-----+
            /     |      \
           v      v       v
.----------. .----------. .------------.
|  Weights | |  scale   | | zero-point |
|   int8   | | float32  | |    int8    |
| constant | | constant | |  constant  |
'----------' '----------' '------------'
</pre>

The dynamic activations on the other hand need to be quantized on-the-fly by inserting the quantization
operations in the graph:

- evaluate the quantization range,
- quantize.

The evaluation of the quantization range is costly because is requires a full-scan of the activations tensor,
which is a bottleneck for parallel processing.

For that reason, the activations quantization ranges are often evaluated before the inference on a selected
number of samples: this is called the calibration of the quantized model.

>Note: the operations that clip their outputs like the bounded ReLU are an exception and don't require an
explicit calibration, since the exact range of their outputs is known in advance.

After calibration, each activation float variable is mapped to an integer variable and two static tensors.

<pre class='diagram'>
               .-----------.
              | Activations |
              |   float32   |
              |  variable   |
              /'-----+-----'\
             /       |       \
            v        v        v
 .-----------.  .----------. .------------.
| Activations | |  scale   | | zero-point |
|   (u)int8   | | float32  | |  (u)int8   |
|  variable   | | constant | |  constant  |
 '-----------'  '----------' '------------'
</pre>

>Note: the activations can be quantized into either `int8` or `uint8`. It is simpler to quantize them to `uint8`
if they correspond to the output of a ReLU operation, since zero-point will be in that case 0.

Conceptually, the resulting graph is a clone of the original graph where all compatible operations are replaced
by a version that operates on tuples of (mantissa, scale, zero-point).

Separating the constant and variable tensors, this leads to the following graphs:

<pre class='diagram'>
              .---------.                   .--------.  .----------. .------------.
             |  Inputs   |                 |  Inputs  | |  scale   | | zero-point |
             |  float32  |                 |  (u)int8 | | float32  | |  (u)int8   |
             | variable  |                 | variable | | constant | |  constant  |
              '----+----'                   '----+---'  '-----+----' '------+-----'
                   |             .               '------------+-------------'
.----------.       v             |\      .----------.         |
| Weights  |   .------.       +--' \     | Weights  |         |
| float32  +->| Matmul |      +--. /     |  int8    +-.       |
| constant |   '---+--'          |/      | constant | |       |         .------------.
'----------'       |             '       '----------' |       |         |   scale    |
                   v                                  |       |       .-+  float32   |
              .---------.                .----------. |       v       | |  constant  |
             |  Outputs  |               |  scale   | |   .-------.   | '------------'
             |  float32  |               | float32  +-+->| QMatMul |<-+
             |  variable |               | constant | |   '---+---'   | .------------.
              '---------'                '----------' |       |       | | zero-point |
                                                      |       |       '-+  (u)int8   |
                                         .----------. |       |         |  constant  |
                                         |zero-point| |       |         '------------'
                                         |  int8    +-'       |
                                         | constant |         |
                                         '----------'         |
                                                              v
                                                          .--------.
                                                         | Outputs  |
                                                         |  (u)int8 |
                                                         | variable |
                                                          '--------'
</pre>



## Quantized linear operations

Most basic Machine Learning operations can be performed using integer arithmetics, which makes them compatible
with linearly quantized inputs.

This does not mean however that one can just replace all floating point operations by an equivalent integer operation:
 the scale and zeropoint of all weights and activations must be taken into account to produce an equivalent graph.

Also, there are two important restrictions with respect to the inputs quantization:
- additions between the integer mantissa of inputs can only be performed if they are in the same scale,
- operations that combine the integer mantissa of inputs channels can only be performed if the channels are in the same scale,\
i.e if the inputs are quantized per-tensor.

>Note: in [another post](/2023/05/quantization-scales-alignment.html) I explain how it is possible to add two inputs quantized with different scales
by adding an explicit alignment operation beforehand.

From an implementation perspective, operations accepting linearly quantized inputs are very specific to each device.

In the next paragraph, I will detail a possible implementation of a quantized matrix multiplication.

## Wrap-up example: a quantized matrix multiplication

Let's consider a simple matrix multiplication of an $X(I, J)$ input by a $W(J, K)$ set of weights:

$Y = X.W$

Since the matrix multiplication multiplies all inputs along the dimension of length $J$ and adds them,
 $X$ cannot be quantized per-axis, because it will lead to the addition of quantized numbers that are not in the same scale.

There is no such restriction on $W$, since the filters along $K$ are all applied independently.

After quantization of the weights per-axis and calibration of the inputs per-tensor, we obtain:

$X \approx X_s * (X_q - X_{zp})$, with $X_s()$, $X_q(I, J)$, $X_{zp}()$

$W \approx W_s * (W_q - W_{zp})$, with $W_s(K)$, $W_q(J, K)$, $W_{zp}(K)$

We can also approximate the outputs per-axis, assuming that the next operation does not require per-tensor inputs.

$Y \approx Y_s * (Y_q - Y_{zp})$, with $Y_s(K)$, $Y_q(I, K)$, $Y_{zp}(K)$

And the quantized graph equivalent to the float operation would be to:

- evaluate the matrix multiplication of the quantized inputs to produce float outputs

$O = X_s * (X_q - X_{zp}) . W_s * (W_q - W_{zp})$

- quantize the float outputs to obtain 8-bit integer outputs

$Y_q = saturate(round(\frac{O}{Y_s}) + Y_{zp})$

- convert back the 8-bit integer outputs to float outputs

$Y \approx Y_s * (Yq - Y_{zp})$

Since $X_s$ is a scalar, and $W_s$ has the same dimension as the outputs last dimension,
the first operation can also be written:

$O = (X_s * W_s) * (X_q - X_{zp}) . (W_q - W_{zp})$

This means that the matrix multiplication can be operated equivalently on integer values,
and the result is a quantized integer number with a scale corresponding to the product of
the inputs and weights scale and a null zero-point.

The quantized sequence of operations is then to:

- evaluate the matrix multiplication of the 8-bit integer inputs to produce n-bit integer outputs

$O_q = (X_q - X_{zp}) . (W_q - W_{zp})$

- convert the n-bit integer outputs to float outputs

$O = (X_s * W_s) * O_q$

- quantize the float outputs to obtain 8-bit integer outputs

$Y_q = saturate(round(\frac{O}{Y_s}) + Y_{zp})$

- convert back the 8-bit integer outputs to float outputs

$Y \approx Y_s * (Yq - Y_{zp})$

The question that should immediately arise at this stage is why we need another quantization
operation after the matrix multiplication, since we already have a quantized output ?

The reason is simply the bitwidth of the outputs: we need an explicit quantization to make
sure that the results of the integer matrix multiplication fit in 8-bit.

>Note: when the operation is followed by a bias addition, the biases are typically quantized to
32-bit with a scale precisely equal to $X_s * W_s$ so that they can be added directly to the outputs
before quantizing.

Going one step further and replacing $O$, since $Y_s$ has the same shape as $X_s * W_s$, we can omit
the third step and write directly:

- evaluate the matrix multiplication of the integer inputs to produce n-bit integer outputs

$O_q = (X_q - X_{zp}) . (W_q - W_{zp})$

- quantize the n-bit integer outputs to obtain 8-bit integer outputs

$Y_q = saturate(round(\frac{X_s * W_s}{Y_s} * O_q) + Y_{zp})$

- convert back the 8-bit integer outputs to float outputs

$Y \approx Y_s * (Yq - Y_{zp})$

This reveals that we can directly quantize the integer outputs of the operation with a scale equal
 to $\frac{Y_s}{X_s * W_s}$.

>Note: Depending on the capabilities of the device, this chain of operations can be implemented in very
different ways on devices that do not have efficient implementations of the integer Matrix Multiplication.

## References

[^quant]: Yunchao Gong, Liu Liu, Ming Yang, Lubomir Bourdev, "Compressing Deep Convolutional Networks using Vector Quantization"
          [arxiv](https://arxiv.org/abs/1412.6115), 2014.

[^qtf]: Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam, Dmitry Kalenichenko,
        "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference"
        [arxiv](https://arxiv.org/abs/1712.05877), 2017.
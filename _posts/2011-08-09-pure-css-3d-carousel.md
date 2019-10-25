---
layout: post
title: Pure CSS 3D Carousel
categories:
- Web Development
tags: Animation Carousel CSS3 3D
published: true
redirect_from: /2011/08/pure-css-3d-carousel/
---
CSS 3D transformations are a wonderful tool to add very impressive animation effects to your web pages without requiring a massive amount of javascript and/or WebGL code.

<!--more-->

More, unlike 2D canvas for instance, if a browser supports CSS 3D Transforms, it is almost certain that it is backed by hardware-acceleration, and thus the effects should not suffer from sluggishness in the animation.

This post demonstrates how a pure 3D CSS carousel can be implemented.

## Design principles

The carousel is constructed dynamically based on the number of cells and their size.

The cells are children elements of a single containing block element.  

    <div class='carousel'>
        <div class='cell>...</div>
        <div class='cell>...</div>
        ...
        <div class='cell>...</div>
        <div class='cell>...</div>
        <div class='cell>...</div>
    </div>

### Defining Cell positions

Relatively to the parent element, each cell element is positioned in the 3D space on an horizontal plane using its [polar coordinates](http://en.wikipedia.org/wiki/Polar_coordinate_system) (ie angle and distance from the coordinates origin).

The <code>N</code> cells are positioned evenly on the carousel using a rotation based on their index <code>n</code>

<math>
  <mrow>
    <mi>rotation</mi>
    <mo>=</mo>
    <mi>n</mi>
    <mo>x</mo>
    <mi>alpha</mi>
  </mrow>
</math>

with 

<math>
  <mrow>
    <mi>alpha</mi>
    <mo>=</mo>
    <mfrac>
     <mrow>
      <mn>2</mn>
      <mo>&#960;</mo>
     </mrow>
     <mi>N</mi>
   </mfrac>
</mrow>
</math>

they are also positioned far enough from the center of the carousel circle so that they don't overlap:

<math>
  <mrow>
    <mi>radius</mi>
    <mo>=</mo>
 <mfrac>
  <mi>width</mi>
  <mrow>
   <mn>2</mn>
   <mo>&#8290;</mo>
   <mrow>
    <mi>tan</mi>
    <mo>&#8289;</mo>
    <mo>(</mo>
    <mfrac>
     <mi>&#960;</mi>
     <mi>N</mi>
    </mfrac>
    <mo>)</mo>
   </mrow>
  </mrow>
 </mfrac>
</mrow>
</math>

The CSS transformation applied to cell n is therefore: 

    .cell:nth-child(n) {
      transform: rotateY((n-1)*alpha) translateZ(radius);
    }

### Configuring the 3D scene

The carousel element is translated in the same way along the Z axis from a negative radius to make sure that all carousel cells are visible. It is also that element that defines the 3D perspective of the scene.

    transform: perspective (1100px) translate(-radiuspx);
    transform-style: preserve-3d;

### Rotating the Carousel

It is only the parent element that is rotated to achieve the carousel effect. 

For instance, to have the Carousel rotated p times to the right, the following additional transformation is applied:

    carousel.style['transform'] = 'perspective (1100px) translate(-radiuspx) rotateY(p*alpha)';

A CSS transition on the Carousel <code>transform</code> property animates the rotation:  

    transition: transform 0.5s;

### Allow customization

To allow the carousel to be specialized for any type of content, the cells are created empty, and it is up to the web page code to populate them using dedicated callbacks specified during initialization.

Specific behaviours can also be attached when a cell enters/leaves the front position.

## Implementation details

The code source of the Carousel is available on [github](https://github.com/kaizouman/3dcarousel) under an MIT licence.

### Dealing with fragmentation
At the time this article is written, CSS 3D Transformations are only available using vendor prefixes (ie -webkit, -moz or -ms).

I decided however to use the standard un-prefixed syntax to keep the code generic and readable.

I also decided to insert generic rules using <code>id</code> and <code>class</code> selectors instead of specifying the transformations on each element using the <code>style</code> attribute.

In order to make it work with existing implementations, I use the [-prefix-free](http://leaverou.github.io/prefixfree/) library to convert the rules on the client-side whenever it is needed: this means that the code will still be valid when an implementation drops vendor prefixes.

The Carousel object includes a special method for that purpose:

    Carousel.prototype.insertRule = function(rule) {
      if( document.styleSheets.length == 0 ) {
        var style = document.createElement('style');
        style.type = 'text/css';
       	document.getElementsByTagName('head')[0].appendChild(style);
      }
      var styleSheet = document.styleSheets[document.styleSheets.length-1];
      // If prefixfree is available, use it
      rule = window.PrefixFree ? PrefixFree.prefixCSS(rule,true):rule;
      // Insert the rule
      styleSheet.insertRule(rule,styleSheet.cssRules.length);  
    } 

### Initialization
The Carousel lives inside a containing element that is passed as a parameter in the Carousel constructor, along with the number of cells and their dimensions:

    container = document.getElementById('container');
    carousel = new Carousel(container,9,465,352, ...)

The other parameters are callback functions to populate the cells and customize the Carousel behaviour.

To disambiguate between multiple instances of Carousel elements that may be inserted in the same page, each one of them is assigned a random ID:
 
    function Carousel(container,nbcell,cwidth,cheight,onadded,onfocus,onblur,onselect){
      this.carousel = document.createElement("div");
      this.id = "Carousel" + Math.floor((Math.random()*10000000)+1);;
      this.carousel.id = this.id;

A specific rule matching the Carousel id is then built and inserted:

    #Carousel8618105 {
        position: relative;
        transform: perspective(1100px) translateZ(-639px);
        transform-style: preserve-3d;
        transition: transform 0.5s;
        ...
    }

Another rule is created to match the class of its children:

    #Carousel8618105 .cell {
        position: absolute;
        left: 0px;
        right: 0px;
        top: 0px;
        bottom: 0px;
        margin: auto;
        ...
    }

Once the carousel has been created, its cell children are added one by one.
 
### Adding cells

As explained in the first paragraph, each cell is a simple div element to which a specific rule is associated using the <code>nth-child</code> selector.

    #Carousel8618105 .cell:nth-child(3) {
        transform: rotateY(80deg) translateZ(639px);
    }

Each cell is created empty: it is up to the caller to populate it using a callback provided at initialization.

### Rotating the Carousel

As explained before, the Carousel is rotated by applying a rotation to the main element.

    Carousel.DIRECTION = {
    LEFT:-1,
    RIGHT:1
    };

    Carousel.prototype.rotate = function(direction) {
      this.blur();
      this.frontIndex = (this.frontIndex - direction + this.nbcell)%this.nbcell;
      this.theta = (this.theta + direction*( 360 / this.nbcell ));
      var style = 'transform: translateZ(-'+this.radius+'px) rotateY(' + this.theta + 'deg)';
      // If prefixfree is available, use it
      style = window.PrefixFree ? PrefixFree.prefixCSS(style,true):style;
      this.carousel.setAttribute('style',style);
      this.focus();
    }

Callback functions specified at intialization can be called when the cell displayed at the front changes. 

## Live Demonstrations

The following demos only work on recent Chrome, Safari or Firefox.

Here is a first example using images.

<div class='picture'>
<a href="/demos/3d-carousel-demo"><img src="/images/3d-carousel-images.png"></a>
<div class='legend'>Click on the image to see the live version</div>
</div>

And another example using this time video elements.

<div class='picture'>
<a href="/demos/3d-video-carousel-demo/"><img src="/images/3d-carousel-videos.png"></a>
<div class='legend'>Click on the image to see the live version</div>
</div>


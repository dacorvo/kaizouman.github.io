---
layout: post
title: 'Different ways of arranging a list of items horizontally'
date: '2013-02-16 19:00:00'
categories:
- Web Development
tags:
- CSS
status: publish
type: post
published: true
redirect_from: /2013/02/different-ways-of-arranging/
---
<nav markdown="1" class='toc'>
<h1>Table of Contents</h1>
* Will be replaced by the TOC
{:toc}
</nav>

This is a very common use case in Web design to arrange a list of items 
horizontally, be it to render a header, a menu, a navigation bar or a footer.

In this article, I will describe three CSS techniques that can be used to
display items horizontally and will try to outline their differences.

<!--more-->

Among the great number of combinations you may want to achieve, I will only 
focus on two layout patterns that are in my opinion the most common ones:

- a centered list of items: you don't care about the relative width of the items as long as they fit on one line in the middle of their container,
- an evenly distributed list of items: you want the items to spread evenly on a single line to take the maximum space horizontally.
 
After a first paragraph detailing the CSS rules that will apply to the three techniques, I will describe in the next paragraphs the following techniques:

- <code>float</code>,
- <code>inline</code>,
- <code>table</code>.

The last paragraph is a quick wrap-up and a tiny bit of advice.

## Common styling

### Base structure

I assume that the elements are organized into an ordered (ol) or unordered
 (ul) list, as it is what makes the most sense in terms of accessibility.

    <ul>
        <li>Lorem</li>
        <li>ipsum</li>
        <li>dolor</li>
        <li>sit</li>
        <li>amet</li>
    </ul>


### Getting rid of user-agent defaults

All browsers provide a default styling for List elements, so our first
task would be to override a few style properties:

    ul.container {
        list-style: none;
        padding-start: 0px;
        margin-before: 0px;
        margin-after: 0px;
    }

Note: a more generic approach to the issue of dealing with browser
default stylesheets is described in 
<a href='http://meyerweb.com/eric/tools/css/reset/'>this post</a> by
<a href='http://meyerweb.com/'>Eric Meyer</a>.

### A better look

Then, we apply a few cosmetic styling rules:

    ul.container {
    	box-shadow: 0 0 5px black;
    }
    ul.container > li {
        text-align: center;
    }
    ul.container > li:nth-child(odd) {
    	background-color: orange;
    }
    ul.container > li:nth-child(even) {
    	background-color: darkorange;
    }


### Shrink to fit

Our first use case requires that our list of items be centered in the page.

Setting the CSS `text-align` property to center on the main containing
block will center all children elements whose width can be computed.

However, the list container being a block element, it will by default
occupy the full width of the page.
  
The solution is to force the browser to evaluate the width of the list
container by applying the 'shrink-to-fit' algorithm described in the 
<a href='http://www.w3.org/TR/CSS21/visudet.html#Computing_widths_and_margins'>
    CSS 2.1 specification
</a>. 

The 'shrink-to-fit' algorithm applies to:

- floated elements,
- absolute positioned elements,
- inside-block elements.

But since we want to center the element, we obviously don't want to use 
float or absolute positioning: we will therefore use an <code>inline-block</code>
list container inside a parent whose <code>text-align</code> property
is set to <code>center</code>.

The basic structure and styling for our use cases is now:

    .centered {
        text-align: center;
    }
    .centered > * {
        display: inline-block;
    }
^
    <p>A centered list of items</p>
    <div class='centered'>
        <ul class='container'>
            <li>Lorem</li>
            <li>ipsum</li>
            <li>dolor</li>
            <li>sit</li>
            <li>amet</li>
        </ul>
    </div>
    <p>A list of items evenly distributed</p>
    <ul class='container'>
        <li>Lorem</li>
        <li>ipsum</li>
        <li>dolor</li>
        <li>sit</li>
        <li>amet</li>
    </ul>    
    <p>Some text after</p>

<style>
div.figure {
    width: 95%;
    border: 1px solid black;
    margin: 0 auto;
    padding: 1em;
}
ul.container {
    list-style: none;
    -webkit-padding-start: 0px;
    -webkit-margin-before: 0px;
    -webkit-margin-after: 0px;
    box-shadow: 0 0 5px black;
}
ul.container > li {
    text-align: center;
}
ul.container > li:nth-child(odd) {
	background-color: orange;
}
ul.container > li:nth-child(even) {
	background-color: darkorange;
}
.centered {
    text-align: center;
}
.centered > * {
    display: inline-block;
}
</style>

Now, let's apply more styling to arrange the items horizontally.

Our target layout is displayed below:

<div id='figure'>
	<img src='/images/posts/different-ways-of-arranging.png'>
</div> 

<!--page-break-->

## Option 1: Float list items to one side

The first option is to use the CSS <code>float</code> property, so that the items 
stack horizontally starting from the side of their container.


    ul.float > li {
        float: left;
    }

However, as we saw in the previous paragraph, the 
<a href='http://www.w3.org/TR/CSS21/visudet.html#Computing_widths_and_margins'>
    CSS 2.1 specification
</a>specifies that floated elements have their width calculated
using a 'shrink-to-fit' algorithm, meaning that we need to specify either
the horizontal padding or the width of the list items to avoid them
wrapping too closely around their content.

For the first use case, we want to allow the items to adjust their width
to the text content, so we will just set the padding: 

    .centered > ul.float > li {
        padding-left: 1em;
        padding-right: 1em;
    }

For the second use case, it is a bit more difficult, as we want the
items to be spread evenly.

Ideally, we would have liked to rely on auto-margins to adjust the
position of each item, but since the items are floated, their auto
computed margins are always set to zero. We therefore need instead to 
set the width of each item explicitly as a fraction of the container
full width. This means in particular that we need to know the exact
number of items in advance.

    ul.float.even > li {
        width: 20%;
    }

Note: here I am using a percentage value, but using a pixel value
is also possible if you know the exact width of the list container 
    
On that subject, we would also need for both use cases to make sure
 that there is enough space horizontally for the items to fit, or 
 the browser will insert a line-break.

<style>
ul.float > li {
    float: left;
}
.centered > ul.float > li {
    padding-left: 1em;
    padding-right: 1em;
}
ul.float.even > li {
    width: 20%;
}
</style>
<div class='figure'>
<p>A centered list of items</p>
<div class='centered'>
<ul class='container float'>
    <li>Lorem</li>
    <li>ipsum</li>
    <li>dolor</li>
    <li>sit</li>
    <li>amet</li>
</ul>
</div>
<p>A list of items evenly distributed</p>
<ul class='container float even'>
    <li>Lorem</li>
    <li>ipsum</li>
    <li>dolor</li>
    <li>sit</li>
    <li>amet</li>
</ul>
<p>Some text after</p>
</div>

As you can see, the rendering is correct for the first use case, but
there are two issues with the second one:

- the element following the list has a reduced vertical margin,
- the list container seems to have disappeared.

The first issue is directly related to the behaviour of floated elements:
 the text following the list tries to wrap around the floated list, but
 since there is no space left, the browser inserts a line-break to 
 continue the wrapping. In other words, the p element following the list
 behaves like an inline element instead of a block element.
 
This issue is typically solved by using the CSS clear property to tell 
the browser when it should stop wrapping elements around the float.

However, as we can see below, it is possible to solve the second issue 
and the first one at once.

### Force the list container to include its children

As a matter of fact, since its children elements have been removed from 
the document flow, the list container has no height unless you specify 
one explicitly.

Although this may not be an issue under some specific circumstances, in 
most cases it does matter:

- you may need (as in our example) to apply a common style to the
container, like a border or a background,
- you may want to associate behaviour to the list container to handle
mouse hovering and clicks.

In those cases, a solution is to use the overflow property to force the 
container to expand vertically:

    ul.float {
        overflow: auto;
    }

<style>
ul.fit {
    overflow: auto;
}
</style>
<div class='figure'>
<p>A centered list of items</p>
<div class='centered'>
<ul class='container float'>
    <li>Lorem</li>
    <li>ipsum</li>
    <li>dolor</li>
    <li>sit</li>
    <li>amet</li>
</ul>
</div>
<p>A list of items evenly distributed</p>
<ul class='container float fit even'>
    <li>Lorem</li>
    <li>ipsum</li>
    <li>dolor</li>
    <li>sit</li>
    <li>amet</li>
</ul>
<p>Some text after</p>
</div>

As a bonus, this also prevents the elements following the list to wrap 
around it, so we don't need to use the CSS clear property.
Two birds killed with one stone !

### Summary of the float technique

Here is the full styling: 

    ul.float {
        overflow: auto;
    }
    ul.float > li {
        float: left;
    }
    .centered > ul.float > li {
        padding-left: 1em;
        padding-right: 1em;
    }
    ul.float.even > li {
        width: 20%;
    }

And markup:

    <p>A centered list of items</p>
    <div class='centered'>
    <ul class='container float'>
        <li>Lorem</li>
        <li>ipsum</li>
        <li>dolor</li>
        <li>sit</li>
        <li>amet</li>
    </ul>
    </div>
    <p>A list of items evenly distributed</p>
    <ul class='container float even'>
        <li>Lorem</li>
        <li>ipsum</li>
        <li>dolor</li>
        <li>sit</li>
        <li>amet</li>
    </ul>
    <p>Some text after</p>

We managed to render our two use cases OK, but there are a two limitations:

- we can only distribute a fixed number of items,
- we need to make sure the container is large enough, 
otherwise the browser will insert a line-break.

<!--page-break-->

## Option 2: Force list items to behave like inline elements

We can force block-level elements to behave like inline-level elements 
by setting the CSS <code>display</code> property to <code>inline</code>
or <code>inline-block</code>. 

    ul.inline > li {
        display: inline;
    }

or

    ul.inline > li {
        display: inline-block;
    }

The only difference between the two values is that when using
<code>display: inline-block</code>, the element retains some of 
its block-level properties, such as the ability to have a width specified.

This may come handy in some configurations, as as mentioned before, 
<code>inline</code> and <code>inline-block</code> elements have their 
width calculated using a 'shrink-to-fit' algorithm, meaning that we need
 to specify either the horizontal padding or the width of the list items
to avoid them wrapping too closely around their content.

In our first use case, we want to let the items adjust to their
text content, so we would only need to specify padding, just as in the
<code>float</code> solution.

    .centered > ul.inline > li {
        display: inline;
        margin-left: 1em;
        margin-right: 1em;        
    }

In the second use case, there is no way to tell the browser that we want
the total width of each item to be calculated evenly, so we will need to
specify it explicitly also (and thus use <code>inline-block</code>):

    ul.inline.even > li {
        display: inline-block;
        width: 20%;        
    }
    
Note: as in the previous technique, we need to make sure that all the 
items can fit on one line. 

Let's look at the resulting layout:

<style>
ul.inline.even > li {
    display: inline-block;
    width: 20%;
}
.centered > ul.inline > li {
    display: inline;
    padding-left: 1em;
    padding-right: 1em;
}
</style>
<div class='figure'>
<p>A centered list of items</p>
<div class='centered'>
<ul class='container inline'>
    <li>Lorem</li>
    <li>ipsum</li>
    <li>dolor</li>
    <li>sit</li>
    <li>amet</li>
</ul>
</div>
<p>A list of items evenly distributed</p>
<ul class='container inline even'>
    <li>Lorem</li>
    <li>ipsum</li>
    <li>dolor</li>
    <li>sit</li>
    <li>amet</li>
</ul>
<p>Some text after</p>
</div>

Ummm, the browser seems to have inserted extra spacing between our menu elements …

### Mind the gap

The gap inserted between the list items is neither margin nor padding:
 it does actually correspond to the space that the browser inserts
 between words inside a text.

Fiddling with padding and margins won't help you fixing this: the only option you have is to remove 
this gap altogether by setting the CSS <code>word-spacing</code> property to a value negating the default
word spacing used by the browser:

    ul.inline {
        word-spacing: -0.3em;
    }
    
Be carefoul though, as this could have nasty side-effects, for instance if you have multiple words in your menu items.

Also, to my knowledge, there is no guarantee that -0.3em will work in all configurations.

<div class='figure'>
<p>A centered list of items</p>
<div class='centered'>
<ul class='container inline' style='word-spacing: -0.3em;'>
    <li>Lorem</li>
    <li>ipsum</li>
    <li>dolor</li>
    <li>sit</li>
    <li>amet</li>
</ul>
</div>
<p>A list of items evenly distributed</p>
<ul class='container inline even' style='word-spacing: -0.3em;'>
    <li>Lorem</li>
    <li>ipsum</li>
    <li>dolor</li>
    <li>sit</li>
    <li>amet</li>
</ul>
<p>Some text after</p>
</div>

### Summary of the inline technique

Here is the full styling: 
    
    ul.inline {
        word-spacing: -0.3em;
    }
    ul.inline.even > li {
        display: inline-block;
        width: 20%;        
    }
    .centered > ul.inline > li {
        display: inline;
        margin-left: 1em;
        margin-right: 1em;        
    }

And markup:

    <div class='figure'>
    <p>A centered list of items</p>
    <div class='centered'>
    <ul class='container inline'>
        <li>Lorem</li>
        <li>ipsum</li>
        <li>dolor</li>
        <li>sit</li>
        <li>amet</li>
    </ul>
    </div>
    <p>A list of items evenly distributed</p>
    <ul class='container inline even'>
        <li>Lorem</li>
        <li>ipsum</li>
        <li>dolor</li>
        <li>sit</li>
        <li>amet</li>
    </ul>
    <p>Some text after</p>
    </div>

There is no real advantage in using this technique over using
the <code>float</code> technique, because we have the same limitations:

- we can only distribute a fixed number of items,
- we need to make sure the container is large enough, 
otherwise the browser will insert a line-break.

In addition, the fiddling with word-spacing complexifies the calculations to achieve the right layout.

<!--page-break-->

## Option 3: Use a CSS Table layout

If you've read carefully through the two previous paragraphs, you may have
noticed that we haven't find so far a solution to automatically spread 
list items evenly in our second use case. 

This is precisely what our second option will bring: let me introduce
<a href="http://www.w3.org/TR/CSS2/tables.html">CSS Tables</a>.

Without going into too much details, and as you may have guessed, CSS 
Tables allow you to define a layout consisting in a 'rectangular grid of
 cells' aka … a table !
 
The idea here will be to define a single-line CSS Table to hold our list
of items.

First, we will assign to the list container and list items the following
display types from the 
<a href="http://www.w3.org/TR/CSS2/tables.html#">CSS Table Model</a>:

    ul.table {
        display: table-row;
    }
    ul.table > li {
        display: table-cell;
    }

Setting explicitly the display mode of the list container is however an 
issue for our first use case, as it will not be centered anymore (it
used to be centered thanks to its <code>inline-block</code> behaviour).
We will therefore need to add an extra <code>div</code> to wrap the 
container:

    <div class='centered'>
    <div>
    <ul class='container table'>
        <li>Lorem</li>
        <li>ipsum</li>
        <li>dolor</li>
        <li>sit</li>
        <li>amet</li>
    </ul>
    </div>
    
Then, similarly as what we did for the two other techniques, we will
need to set the list items padding explicitly as the <code>table-cell</code>
display-mode also uses the shrink-to-wrap algorithm.

    .centered ul.table > li {
        padding-left: 1em;
        padding-right: 1em;    
    }
  
Now, for the second use case, we will take advantage of the CSS Tables
<a href='http://www.w3.org/TR/CSS2/tables.html#fixed-table-layout'>
Fixed Table Layout algorithm
</a>
to force the horizontal layout to be calculated so that the table fills
the whole width of its container regardless of each cell content.

    ul.table.even {
        table-layout: fixed;
        width: 100%;
    }

Note: one big difference with the previous two techniques is that we don't need
to take care of the line width, as the items will be rendered on a single
line even if they don't fit in their container.

<style>
ul.table {
    display: table;
}
ul.table > li {
    display: table-cell;
}
.centered ul.table > li {
    padding-left: 1em;
    padding-right: 1em;    
}
ul.table.even {
    width: 100%;
    table-layout: fixed;
}
</style>
<div class='figure'>
<p>A centered list of items</p>
<div class='centered'>
<div>
<ul class='container table'>
    <li>Lorem</li>
    <li>ipsum</li>
    <li>dolor</li>
    <li>sit</li>
    <li>amet</li>
</ul>
</div>
</div>
<p>A list of items evenly distributed</p>
<ul class='container table even'>
    <li>Lorem</li>
    <li>ipsum</li>
    <li>dolor</li>
    <li>sit</li>
    <li>amet</li>
</ul>
<p>Some text after</p>
</div>

### Summary of the CSS table technique

Here is the full styling: 

    ul.table {
        display: table;
    }
    ul.table > li {
        display: table-cell;
    }
    .centered ul.table > li {
        padding-left: 1em;
        padding-right: 1em;    
    }
    ul.table.even {
        width: 100%;
        table-layout: fixed;
    }

and markup:

    <p>A centered list of items</p>
    <div class='centered'>
    <div>
    <ul class='container table'>
        <li>Lorem</li>
        <li>ipsum</li>
        <li>dolor</li>
        <li>sit</li>
        <li>amet</li>
    </ul>
    </div>
    </div>
    <p>A list of items evenly distributed</p>
    <ul class='container table even'>
        <li>Lorem</li>
        <li>ipsum</li>
        <li>dolor</li>
        <li>sit</li>
        <li>amet</li>
    </ul>
    <p>Some text after</p>
    
This solution allows more flexibility thant the previous ones:

- we can spread an arbitrary number of items,
- the items will stay in line whatever the width of their container is.

It requires however an extra wrapping <code>div</code> to address the 
first use case. 

<!--page-break-->

## To conclude

In the table below, I have summarized the differences between the three techniques.
<style>
table {
	max-width: 95%;
	text-align: center;
}
table,tr,td {
	border: 1px solid;
	border-collapse: collapse;
}
tr:first-child {
	background-color: lightgrey; 
}
td {
	width: 8em; 
}
</style>
<table>
<tr>
<td>Technique</td>
<td>Center items</td>
<td>Spread items (fixed)</td>
<td>Spread items (arbitrary)</td>
<td>Line-break Hazard</td>
</tr>
<tr>
<td>float</td>
<td>Yes</td>
<td>Yes</td>
<td>No</td>
<td>Yes</td>
</tr>
<tr>
<td>inline</td>
<td>Yes</td>
<td>Yes</td>
<td>No</td>
<td>Yes</td>
</tr>
<tr>
<td>table</td>
<td>Yes</td>
<td>Yes</td>
<td>Yes</td>
<td>No</td>
</tr>
</table>

All three methods allow you to achieve the target rendering for both use cases.

If you can live with their limitations, you can use either one of the first two (although I have a mild preference for the <code>float</code> technique).

The most flexible technique, and the one providing the most predictible results is however the <code>table</code> technique, and that would be my recommendation for these two use cases.

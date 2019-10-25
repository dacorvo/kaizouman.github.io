---
layout: post
title: 'Delaunay Triangulation applied in real-time on a WebCam stream'
author: 'David Corvoysier'
date: '2013-05-10 22:00:00'
categories:
- Web Development
tags:
- Video
- WebCam
- getUserMedia
- Canvas
type: post
published: true
redirect_from: /2013/05/delaunay-effect-webcam/
---
I already detailed in a [previous post](/2012/09/frame-by-frame-video-effects-using-html5-and/) how typical image processing algorithms could be applied in real-time on a video stream using the HTML5 <code>canvas</code> to produce video effects.
 
In this article, I will explain how the same kind of effects can be applied on a WebCam stream thanks to the <code>getUserMedia</code> API.

<!--more-->

## Capturing WebCam frames in the Canvas

It is not possible (yet ?) to grab directly a WebCam frame to inject it in the HTML5 <code>canvas</code>: you have to go through an intermediary step, capturing first the stream in a <code>video</code> element, then transferring it frame by frame in the <code>canvas</code>.

As displayed below, we will use the [<code>getUserMedia</code>](http://dev.w3.org/2011/webrtc/editor/getusermedia.html) API to grab the video stream from a WebCam and inject it in a <code>video</code> element:

	navigator.getUserMedia({video: true}, 
		function(stream) {
			video.src = URL.createObjectURL(stream);
			// Process frames here
			...
		}, 
		function (error) {
			// Error handling
		});

When the <code>getUserMedia</code> API is called, the user will be prompted to give access to its WebCam:

* acceptance will trigger the first function that creates an URL object from the stream and pass it to the video element,
* refusal will trigger the error function.

Once the stream has been successfully redirected to the <code>video</code> element, we will start capturing frames in the <code>canvas</code> at regular intervals timed by <code>requestAnimationFrame</code>. To make sure that there is actually something to be captured, we test against the <code>HAVE_ENOUGH_DATA</code> state for the <code>video</code> element before grabbing a frame. 

	function tick() {
	
		requestAnimationFrame(tick);
		
		if (video.readyState === video.HAVE_ENOUGH_DATA) {
			...
			ctx.drawImage(video, 0, 0, canvasWidth, canvasHeight);
		var imageData = ctx.getImageData(0, 0, cwidth, cheight);
			...
                    
Once a frame has been transferred to the <code>canvas</code>, it is extracted as a byte array to apply our image processing algorithms.
 
## Detecting the singularity points

We will now use a computer vision algorithm called [FAST](http://www.edwardrosten.com/work/fast.html)to detect remarkable points ("corners") in the image. To ease corner detection, the frame has first to be converted to a grayscale image.

Several Open Source implementations of the corresponding algorithms exist on the Web: we will use the [JSFeat](http://inspirit.github.io/jsfeat/) library that provides a neat wrapper around optimized implementations of the most typical ones, including grayscale and FAST.
    
	jsfeat.imgproc.grayscale(imageData.data, img_u8.data);
	jsfeat.fast_corners.set_threshold(threshold)
	jsfeat.fast_corners.detect(img_u8, corners, 5);

The corners detected in the image are stored in an array of points: {x,y}.

## Applying Delaunay Triangulation

We then apply a [Delaunay Triangulation](http://en.wikipedia.org/wiki/Delaunay_triangulation) alogithm to the set of points to identify triangles covering the image.

    var triangles = triangulate(vertices);
    
We will use at this stage a fast Open Source implementation of the Delaunay Triangulation algorithm developed by [ironwallaby](https://github.com/ironwallaby/delaunay).

## Rendering back to Canvas

The final step is to render back the result of the Delaunay Triangulation to the canvas by painting each triangle with a color representing its content in the original image.

For the sake of simplicity, we will just pick the color of a point that we know to be inside the triangle, but a more complex process could be used to select a real average of the triangle colors.

	var getTriangleColor = function (img,triangle) {
		var getColor = function (point) {
			var offset = (point.x+point.y*cwidth)*4;
			return {    r:img.data[offset],
						g:img.data[offset+1],
						b:img.data[offset+2]  };
		}
		var midPoint = function (point1,point2) {
			return {x:(point1.x+point2.x)/2,
					y:(point1.y+point2.y)/2};
		}
		// Pick a point inside the triangle
		var point1 = midPoint(triangle.a,triangle.b);
		var point = midPoint(point1,triangle.c);
		return getColor({x:Math.floor(point.x),y:Math.floor(point.y)});
	}

Each triangle is painted using simple canvas drawing primitives:

	for(var i=0;i<triangles.length;i++) {
		var color = triangles[i].color = getTriangleColor(imageData,triangles[i]);
		gridCtx.fillStyle = 'rgb('+
			color.r+','+
			color.g+','+
			color.b+')';
		gridCtx.beginPath();
		gridCtx.moveTo(triangles[i].a.x,triangles[i].a.y);
		gridCtx.lineTo(triangles[i].b.x,triangles[i].b.y);
		gridCtx.lineTo(triangles[i].c.x,triangles[i].c.y);
		gridCtx.closePath();
		gridCtx.fill();
	}


## Demo

Click on the image below to see how it works when all the pieces are put together:

<div class='picture'>
<a href='http://kaizouman.github.io/js-delaunay-effect/'>
<img src='/demos/delaunay-effect/delaunay-effect.jpg'>
</a>
</div>


The code for this demo is [available on github](https://github.com/kaizouman/js-delaunay-effect).



 

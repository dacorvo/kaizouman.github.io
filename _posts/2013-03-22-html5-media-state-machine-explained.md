---
layout: post
title: 'Media element state machine explained'
author: 'David Corvoysier'
categories: Video
tags: HTML5 video
published: true
redirect_from: /2013/02/html5-media-state-machine-explained/
---

The HTML5 specification gives a detailed description of the
[algorithm](http://www.w3.org/TR/html5/embedded-content-0.html#loading-the-media-resource) 
to be applied when rendering a media content.

The text is however sometimes a bit cryptic, with a lot of details only 
relevant to implementers. I will try in this article to give a simplified
description of the expected behaviour of a Media element from a web 
application developer perspective. 

<!--more-->

One may distinguish typically three phases:

* initialization: the Media element selects a media source and retrieves
 its properties (duration, size)
* buffering: the Media element retrieves and store as many data as 
required to start rendering the content 
* playback: the Media element decodes and renders the content 

<div class='figure'>
<img src='/images/HTML5MediaStateDiag.svg'>
</div>

# Initialization

A newly inserted HTML Media element will be initialized only:

* when its <code>src</code> attribute is set,
* when a children <code>source</code> element is inserted.

Note: both situations can occur either declaratively (through markup) or 
programmatically (through javascript).

An initialized HTML Media element will be reset only:

* when its <code>src</code> attribute is modified,
* when its <code>load()</code> method is invoked.

Note: An HTML Media element will not be reset when a <code>source</code>
child is inserted or modified. 

During the initialization phase, the user-agent will apply the 
[Media resource selection algorithm](http://www.w3.org/TR/html5/embedded-content-0.html#concept-media-load-algorithm).
to select the most appropriate media resource. 

At the end of the initialization phase, the Media element should have:

* its <code>networkState</code> set to <code>NETWORK_LOADING</code>,
* its <code>readyState</code> set to <code>HAVE_NOTHING</code>,
* its <code>currentSrc</code> set to the actual Media source URL.

The algorithm may be blocked at this stage until an explicit user request
 to play the content. This happens in particular:

* if the Media element <code>preload</code> attribute has been set to 
<code>none</code>,
* on some user-agents (typically apple mobile devices) that want to 
prevent the user to be charged for useless data transfer
(Please refer to [this article for details](http://developer.apple.com/library/safari/#documentation/AudioVideo/Conceptual/Using_HTML5_Audio_Video/Device-SpecificConsiderations/Device-SpecificConsiderations.html#//apple_ref/doc/uid/TP40009523-CH5-SW1)).

Otherwise, the browser will immediately transition to the buffering phase.

# Buffering

During the buffering, phase, the user-agent will fetch the selected
Media resource as described in the
[Media resource fetch algorithm](http://www.w3.org/TR/html5/embedded-content-0.html#concept-media-load-resource).  

If the <code>autoplay</code> attribute of the <code>media</code> element
is set to <code>true</code>, or if the <code>play()</code> method has
been called explicitly, the user-agent will immediately try to download 
as much data as needed to play the content through.
Otherwise, the amount of data loaded at this stage is mostly implementation
dependent.

The application developer may however influence the browser preloading
behaviour by setting the <code>preload</code> attribute to:

* <code>metadata</code> to download only what is needed to determine the
 duration and dimension of the content,
* <code>auto</code> to download proactively as much data as needed to 
be able to start playback immediately.

At the end of the buffering phase, the Media element could have its
<code>readyState</code> set to:

* <code>HAVE_META_DATA</code>: only the duration and the dimension of the
content are then available. At this stage, the browser will have sent a 
<code>loadedmetadata</code> event.
* <code>HAVE_CURRENT_DATA</code>: a single frame of content is available
(and can be used to be displayed in a canvas for instance). At this stage,
 the browser will have sent in addition a <code>loadeddata</code> event.
* <code>HAVE_FUTURE_DATA</code>: enough frames are available to start 
playback. At this stage, the browser will have sent in addition a 
<code>canplay</code> event.
* <code>HAVE_ENOUGH_DATA</code>: enough frames are available to play the
content through. At this stage, the browser will have sent in addition a 
<code>canplaythrough</code> event.

If the <code>autoplay</code> attribute of the Media element is set to 
<code>true</code>, the browser will wait until it reaches the 
<code>HAVE_ENOUGH_DATA</code> state to transition to the next step and 
start rendering content.

However, if the <code>play</code> method has been called explicitly,
the playback will start as soon as the <code>HAVE_FUTURE_DATA</code> 
state is reached. 

# Playback

As mentioned in the previous paragraph, the Media element cannot start 
playing content before having reached at least the <code>HAVE_FUTURE_DATA</code>
 state that corresponds to the <code>canplay</code> event.

The user-agent constructs a media timeline based on the metadata 
retrieved from the stream.
In most case, it will be the timeline as described in the original stream,
with the following exceptions:

* if the Media resource specifies an explicit start date, the user-agent
will store it in the <code>startDate</code> attribute, but will define its
timeline relatively to it, starting from zero.
* if the Media resource specifies a discontinuous timeline, the user-agent
will expand the timeline of the first content to the entire stream.

The application developer can specifiy an initial playback position by either:

* setting the Media element <code>currentTime</code> attribute to that
position,
* specifying the position using a Media fragment URI.

If no specific playback position has been specified, the user-agent will
 start the playback at the initial playback position defined in the stream.
 
During playback, the browser exposes the 'official' playback position in the
<code>currentTime</code> attribute, that doesn't necessarily reflect 
accurately the real playback position.

The speed at which the content is being played is exposed by the user-agent
in the <code>playbackRate</code> attribute of the Media element.

Unless specified differently in the <code>defaultPlaybackRate</code> 
attribute, a content will be initially played at an 1.0 rate.

If at any time during playback the Media element runs out of data, it will generate a <code>waiting</code> event and switch back to the buffering phase. 

When the end of the stream is reached:

* the <code>ended</code> attribute is set to <code>true</code>,
* an <code>ended</code> event is emitted,
* if the <code>loop</code> attribute is set to true, the playback resumes
at the earliest playback position: otherwise it stops.


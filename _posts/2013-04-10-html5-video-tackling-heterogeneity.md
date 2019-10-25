---
layout: post
title: 'Tackling the heterogeneity of HTML5 Video'
author: 'David Corvoysier'
date: '2013-04-10 12:00:00'
categories:
- Web Development
tags: HTML5 video heterogeneity
published: true
redirect_from: /2013/04/html5-video-tackling-heterogeneity/
---
The HTML5 Media Elements specification has reached a level of maturity 
that would allow video to be delivered in a Web browser with the same 
level of features as when using a plugin. 

However, partly due to historical reasons and partly due to the increased
diversity of implementations and device form factors, you have to overcome
several hurdles if you want to deliver cross-platform video content.

<!--more-->

## Providing alternative sources for the same content

The HTML5 video tag supports multiple Media sources to be specified for 
a specific content, letting the browser decide the one that is
 the most appropriate based on the 
[Media resource selection algorithm](http://www.w3.org/TR/html5/embedded-content-0.html#concept-media-load-algorithm).
 
Alternative media resources for a single multimedia content are specified
using the <code>source</code> element as children of the <code>video</code>
element.

The <code>source</code> element has two attributes that are used by the
browser to select the appropriate resource:

* the <code>type</code> attribute defines the Media format of the content,
* the <code>media</code> attribute can be used by the service to describe
the device the resource is intended for using the 
[Media Query](http://dev.w3.org/csswg/css3-mediaqueries/) syntax.

The <code>type</code> attribute comprises a mandatory MIME type and an
optional <code>codecs</code> parameters using the syntax described in
[RFC4281](http://tools.ietf.org/html/rfc4281).

The type itself attribute is not mandatory, but if it is not provided, the 
browser will have to download at least the few first bytes of each source to 
identify the media encoding and verify if it can play it. Specifying at 
least the content MIME-type is therefore recommended to save bandwidth.

Although it would probably be overkill for most web sites, further specifying
 the content encoding using the codecs parameters may nevertheless be useful
 for a commercial-grade service.

In the example below, three alternative resources are provided with an 
increasing level of video complexity (baseline, main, high): 

    <video>
      <source src='video_baseline.mp4' 
              type='video/mp4; codecs="avc1.42E01E, mp4a.40.2"'>
      <source src='video_main.mp4' 
              type='video/mp4; codecs="avc1.4D401E, mp4a.40.2"'>
      <source src='video_extended.mp4' 
              type='video/mp4; codecs="avc1.64001E, mp4a.40.2"'>
    </video>

Unfortunately, the <code>codecs</code> parameter is limited to the 
description of the codecs, and cannot be used to describe Media features,
such as spatial resolution. It is however possible to work around this
limitation using the <code>media</code> attribute.

The <code>media</code> attribute comprises a type parameter
followed by several media expressions. Although several types have been defined
 in the legacy HTML and CSS specifications, only <code>all</code>,
 <code>screen</code> and <code>print</code> are actually supported.

>Note: It is in particular not advised to use the <code>handheld</code> type
> to identify a mobile device. The widely supported <code>device-width</code>
> and <code>device-height</code> expressions are more appropriate. 

The example below specifies two alternative contents based on the screen size: 

    <video>
      <source src='video_small.mp4' media='screen and (max-device-width:800px)'>
      <source src='video_big.mp4' media='screen and (min-device-width:801px)'>
    </video>

## Selecting the proper Media formats

Due to the lack of consensus on this subject, the HTML5 specification 
doesn't mandate any specific audio or video format: it is up to the
user-agent (ie the browser) to define which format should be supported,
the decision being mainly driven by licensing terms.

As of today, there are still two competing sets of Media formats:

* MP4/H264/AAC
* WebM/VP8/Vorbis

In the past, there was a clear split between browser vendors, with Apple
 and Microsoft backing MP4/H264/AAC (for which they have patents) and facing
 a strong opposition coming from Opera and Firefox, Chrome mostly remaining
 neutral on the subject.

The situation has evolved a bit, since H264/AAC decoding is often either
 supported by the underlying hardware (especially on mobile chipsets), or
 a system-wide multimedia framework (like Media Foundation for Windows or
 gstreamer for Linux), thus mitigating the licensing issues. 
 
Firefox therefore [now supports](https://developer.mozilla.org/en-US/docs/HTML/Supported_media_formats)
what they call 'patents-encumbered' media formats if they are already 
available on the system. 

In the meantime, VP8 failed to get a real momentum, probably due to its
lack of proven improvements towards H264.

As a consequence, the most sensible option today is to choose MP4/H264/AAC 
as the main codec combination for encoding your content, as it 
has [the widest level of support](http://caniuse.com/#feat=mpeg4).            

You may also consider providing an alternate version encoded using WebM.

    <video>
      <source src='sintel.mp4' type='video/mp4'>
      <source src='sintel.webm'type='video/webm'>
    </video>

Tools like [handbrake](http://www.handbrake.fr) may be used to convert from one format to the other. 

## Adapting content to the target device

Event if you restrict yourself to a single combination of container and
codecs, it is highly recommended to be able to adapt the video content you
deliver to the device that will render it.

You may for instance consider specifying several qualities for a specific content, with an increasing complexity and spatial resolution.

When identifying sources of different qualities, you can use detailed codecs parameters and/or a media query:

    <video>
      <source src='video_low.mp4' 
              type='video/mp4; codecs="avc1.42E01E, mp4a.40.2"'
              media='screen and (max-device-width:480px)'>
      <source src='video_std.mp4'
              type='video/mp4; codecs="avc1.4D401E, mp4a.40.2"'
              media='screen and (min-device-width:480px) and (max-device-width:1280px)'>
      <source src='video_hi.mp4'
              type='video/mp4; codecs="avc1.64001E, mp4a.40.2"'
              media='screen and (min-device-width:1280px)'>
    </video>

You can find various encoding recommendations on the web to address multiple
 devices. This [article](http://knowledge.kaltura.com/best-practices-multi-device-transcoding)
 provides a detailed list of encoding profiles for desktop, mobile and
 other embedded devices.

## Beware of bogus Web-server MIME-types for media content

As mentioned before, when specifying a Media source, the type parameter can be omitted.

In that case, the browser will download the first few bytes of each source to figure out its encoding, starting from the MIME-type specified by the Web server.

However, if the Web server is not configured properly, it may declare the wrong mime-type for the content: webm content would typically be sent as text/plain for instance.

Below is an example of the lines that can be added to an Apache .htaccess to identify video content based on the file extension:

    AddType video/mp4 .mp4
    AddType video/mp4 .m4v
    AddType video/webm .webm
    AddType audio/webm .weba

And that's pretty much about it for now. 

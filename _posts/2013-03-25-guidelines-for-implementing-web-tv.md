---
layout: post
title: 'Guidelines for implementing a Web TV Service'
author: 'David Corvoysier'
categories: TV
tags: HTML5 video
published: true
redirect_from: /2013/03/guidelines-for-implementing-web-tv/
---

The HTML5 specification has reached a level of maturity that allows TV
services to be delivered in a Web browser. This article provides a set
of guidelines to implement a typical TV service using Web Technologies,
and gives details about the level of support to be expected for each 
feature.

<!--more-->

The HTML5 video tag allows audio and video files to be rendered directly
by the browser, although most implementations will actually delegate most
of the multimedia processing to underlying components.

The HTML5 video is supported by all recent desktop and mobile browsers.
 Please refer to [caniuse/video](http://caniuse.com/#feat=video) or
 ["The State Of HTML5 Video"](http://www.longtailvideo.com/html5/) for details.

The following paragraphs provides a description of the Media elements features that are the most relevant from a TV service perspective, focussing on the features included in the HTML 5.0 specifications.

>Newer specifications are being developed at the time this article is written, and will not be detailed here:
>
>* [Media Source Extensions](http://www.w3.org/TR/media-source/): allows fine-grained control over the media pipeline to perform adaptive streaming or advert insertion from javascript
>* [Encrypted Media Extensions](https://dvcs.w3.org/hg/html-media/raw-file/tip/encrypted-media/encrypted-media.html): adds support for Digital Rights Management (very controversial)
>
>Even though early implementations already exist in Google Chrome, these specifications are not mature yet and it is too early to rely on them to develop a mainstream TV service, unless you are able to control both the user-agent and the server.

# Controls

Setting the <code>controls</code> properties of the Media element to true will activate the user-agent native multimedia controls.

It is very likely however that a web TV application will require a level of interaction with the user leading to overriding at least some of the default behaviors assumed by these controls, that would therefore only be useful in debug mode.

# Media formats

Due to the lack of consensus on this subject, the HTML5 specification 
doesn't mandate any specific audio or video format: it is up to the
user-agent (ie the browser) to define which format should be supported,
the decision being mainly driven by licensing terms.

As of today, there is still two competing set of Media formats:

* MP4/H264/AAC
* WebM/VP8/Vorbis

In the past, there was a clear split between browser vendors, with Apple
 and Microsoft backing MP4/H264/AAC (for which they have patents) and facing
 a strong opposition coming from Opera and Firefox, Chrome mostly remaining
 neutral on the subject.

The situation has evolved a bit, since H264/AAC decoding is often either
 supported by the underlying hardware (especially on mobile chipsets), or
 a system-wide multimedia framework (like gstreamer for instance), thus 
 mitigating the licensing issues. 
 
Firefox therefore [now supports](https://developer.mozilla.org/en-US/docs/HTML/Supported_media_formats)
what they call 'patents-encumbered' media formats if they are already 
available on the system. 

In the meantime, VP8 failed to get a real momentum, probably due to its
lack of proven improvements towards H264.

As a consequence, the most sensible option today is to choose MP4/H264/AAC 
as the main (only ?) codec combination for encoding your content, as it 
has [the widest level of support](http://caniuse.com/#feat=mpeg4).            

# Adapting content to the target device

Event if you restrict yourself to a single combination of container and
codecs, it is highly recommended to be able to adapt the video content you
deliver to the device that will render it.

The HTML5 video tag supports multiple Media sources to be specified for 
a specific content, and it is up to the browser to select the one that is
 the most appropriate based on the 
[Media resource selection algorithm](http://www.w3.org/TR/html5/embedded-content-0.html#concept-media-load-algorithm).
 
You can find various encoding recommendations on the web to address multiple
 devices. This [article](http://knowledge.kaltura.com/best-practices-multi-device-transcoding)
 provides a detailed list of encoding profiles for desktop, mobile and
 other embedded devices.

The list of encoding you would typically need to support is to be defined
on a service basis, but as a rule of thumb, for a generic purpose TV
service, supporting at least the three 'standards' resolutions is recommended:

* Low Definition: 480x360
* Standard Definition: 1280x720
* High Definition: 1980x1080

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

It is recommended to use the <code>codecs</code> parameter to explicitly
specify the audio and video codecs of a specific resource.

In the example below, three alternative resources are provided with an 
increasing level of video complexity (baseline, extended, main): 

{% highlight html %}
    <video>
      <source src='video.mp4' 
              type='video/mp4; codecs="avc1.42E01E, mp4a.40.2"'>
      <source src='video.mp4' 
              type='video/mp4; codecs="avc1.58A01E, mp4a.40.2"'>
      <source src='video.mp4' 
              type='video/mp4; codecs="avc1.4D401E, mp4a.40.2"'>
    </video>
{% endhighlight %}

Unfortunately, the <code>codecs</code> parameter is limited to the 
description of the codecs, and cannot be used to describe Media features,
such as spatial resolution. It is however possible to work around this
limitation using the <code>media</code> attribute.

The <code>media</code> attribute comprises a <code>type</code> parameter
followed by several media expressions.

Although several types have been defined in the legacy HTML and CSS
specifications, only <code>all</code>, <code>screen</code> and 
<code>print</code> are actually supported. 

It is in particular not advised to use the <code>handheld</code> type to
 specify that a specific Media resource is intended to be rendered on a 
 mobile device: use instead the widely supported <code>device-width</code>
  and <code>device-height</code> based expressions: 

{% highlight html %}
    <video>
      <source src='video.mp4' 
              type='video/mp4; codecs="avc1.42E01E, mp4a.40.2"'
              media='screen and (max-device-width:480px)'>
      <source src='video.mp4'
              type='video/mp4; codecs="avc1.58A01E, mp4a.40.2"'
              media='screen and (min-device-width:480px) and (max-device-width:1280px)'>
      <source src='video.mp4'
              type='video/mp4; codecs="avc1.4D401E, mp4a.40.2"'
              media='screen and (min-device-width:1280px)'>
    </video>
{% endhighlight %}

# Content playback

A content can be set to play automatically by setting the <code>autoplay</code> attribute to <code>true</code>.

{% highlight html %}
    <video src="video.mp4" autoplay></video>
{% endhighlight %}

Alternatively, the application can call explicitly the <code>play</code> method of the Media element:

{% highlight javascript %}
    video.play();
{% endhighlight %}

> Special case: Unlock playback on iOS devices
>
> On iOS devices were the user may be on a cellular network, no data can be fetched from the network until the user initiates it (Please refer to [this article for details](http://developer.apple.com/library/safari/#documentation/AudioVideo/Conceptual/Using_HTML5_Audio_Video/Device-SpecificConsiderations/Device-SpecificConsiderations.html#//apple_ref/doc/uid/TP40009523-CH5-SW1)).
>
> Unlocking the playback can then only be achieved:
>
> * through the video native controls,
> * by calling the <code>play()</code> method in a user event callback (ie a key, touch or mouse event). 

During playback, the user-agent provides feeback to the application by:

* updating the <code>currentTime</code> attribute,
* generating <code>timeupdate</code> events at regular intervals (typically 15 to 250ms). 

If no specific playback position has been specified, the user-agent will start the playback at the initial playback position defined in the stream.

The application can seek programmatically in the media timeline by setting the <code>currentTime</code>
attribute to a new playback position:

{% highlight javascript %}
    video.currentTime = 10;
{% endhighlight %}

Alternatively, a playback position can be specified declaratively using a Media fragment URI:

{% highlight html %}
    <video src="video.mp4#t=10'></video>
{% endhighlight %}

For some Media resources, seeking may be limited to only some parts of 
the content: the user-agent therefore exposes the time ranges to when it
 is possible to seek through the <code>seekable</code> attribute of the 
 Media element.
The start of the first element in the seekable time range represents the
 earliest possible playback position.
 
Example: a live broadcast content would always have an earliest playback 
position that is equal to the current playback position.

Trick-modes are achieved by altering the value of the <code>playbackRate</code>
attribute:
 
* normal: rate = 1.0
* pause: rate = 0
* slow-forward: 0 < rate < 1.0
* fast-forward: 1.0 < rate
* slow-rewind: -1.0 <= rate < 0
* fast-rewind: rate < -1.0

When playing a content backwards, the audio is muted. It may also be muted
by the user-agent when playing a content forward at a rate that is not 1.0.

If the earliest playback position is reached when playing backwards, the 
playback stops.

# Buffering strategies

This paragraph assumes that the reader is familiar with the 
[Media element state machine](/2013/03/22/html5-media-state-machine-explained).

When presenting multimedia content for playback, a TV application must find the best buffering strategy to address two contradictory user expectations:

* to be able to watch the content as soon as possible ("low latency"),
* to be able to watch the content without any interruption ("play through").

In addition, the TV application will usually want to optimize network bandwidth and memory consumption, typically by avoiding unnecessary downloads.

## Control buffering before playback

By default, the user-agent will apply an automatic strategy to aggressively preload a content as soon as a valid source has been identified for the presented media.

The HTML5 Media element however exposes a <code>preload</code> attribute to allow the web application to define the amount of data that can safely be preloaded by the user-agent before the content playback is explicitly started. 

Note: The value of the <code>preload</code> attribute is ignored when a content is in autoplay.  

By setting the <code>preload</code> attribute to <code>none</code>, the application can prevent the user-agent from downloading any data before the Media element is explicitly requested to play the content.

Setting the <code>preload</code> attribute to <code>metadata</code> will tell the user-agent to download only the amount of data required to identify the content duration and dimensions.

Reverting the <code>preload</code> attribute to its default <code>auto</code> value will tell the user-agent to aggressively preload the content, as if the content was about to be played. 

Note: in terms of buffering, setting <code>preload</code> to <code>auto</code> before requesting a content to play is equivalent to calling the <code>play</code> method directly. 

As a rule of thumb:

* all contents should be inserted by default with <code>preload</code> explicitly set to <code>none</code>,
* the contents that are likely to be played should have <code>preload</code> set to <code>metadata</code>,

The <code>preload</code> attribute can also be used once playback has started, as explained in the next paragraph.

## Control buffering during playback

During playback, the <code>preload</code> attribute allows the web application to control how much data is being buffered in advance:

* setting <code>preload</code> to <code>auto</code> allows the user-agent to aggressively download content, up to having it entirely stored in memory,
* setting <code>preload</code> to <code>metadata</code> tells the user-agent to limit its internal buffers to the amount of data required to play the content without interruption.

The amount of data buffered can be queried using the <code>buffered</code> attribute, allowing the application to dynamically adjust <code>preload</code> for a finer grained-control over the buffering policy.

A application willing to limit buffering during playback would typically:

* start with <code>preload = auto</code>,
* on download <code>progress</code>, set <code>preload = metadata</code> if we are above the buffering threshold,
* on playback <code>timeupdate</code> events, set <code>preload = auto</code> if we are below the buffering threshold.

See example code below:

{% highlight javascript %}
    function getBufferedRange(video) {
        var i = video.buffered.length - 1;
        while((i>0) && 
              (video.buffered.start(i)>video.currentTime) {
            i--;
        }
        return (video.buffered.end(i) - video.currentTime);
    }
    video.ontimeupdate = function (e) {
        if (getBufferedRange(video) < THRESHOLD){
            video.preload = 'auto';            
        }
    }
    video.onprogress = function (e) {
        if (getBufferedRange(video) >= THRESHOLD){
            video.preload = 'metadata';            
        }
    }
{% endhighlight %}

## Avoid interruptions in playback

The default behaviour of a Media player in 'autoplay' mode is to wait until enough data to be able to play the content through has been retrieved before starting rendering a content on the screen. 

Using the same terminology as the HTML5 specification: in autoplay mode, the playback doesn't start until the <code>HAVE_ENOUGH_DATA</code> state has been reached. 

There is therefore no specific configuration to apply to achieve that behaviour but to set the <code>autoplay</code> attribute to <code>true</code>.

{% highlight html %}
    <video src='video.mp4' autoplay></video>
{% endhighlight %}

An alternative would be to listen to the <code>canplaythrough</code> event and call the <code>play</code> method explicitly. 

{% highlight javascript %}
    video.addEventListener('canplaythrough',
	function (e) {
		play();
	},false);
{% endhighlight %}

## Minimize latency

The playback of a content cannot start before the Media player has received enough data to decode at least a few frames.

Using the same terminology as the HTML5 specification: the playback cannot start before the <code>HAVE_FUTURE_DATA</code> state has been reached.

In order to start the playback of a content as soon as possible, a web application can detect the transition to the  <code>HAVE_FUTURE_DATA</code> state by listening to the <code>canplay</code> event, and call the <code>play</code> explicitly:

{% highlight javascript %}
    video.addEventListener('canplay',
	function (e) {
		play();
	},false);
{% endhighlight %}

# In-band tracks

The HTML5 Media element supports multiple inband tracks for a specific media content:  for example, in addition to the primary video and audio tracks, a media resource could have foreign-language dubbed dialogues, director's commentaries, audio descriptions, alternative angles, or sign-language overlays.

## In-band media tracks

Inband media tracks are exposed through the <code>audioTracks</code> and <code>videoTracks</code> attributes, and would become available as soon as the Media player has reached the <code>HAVE_META_DATA</code> state.

To select programmatically a specific media track, an application would thus typically listen to the <code>loadedmetadata</code> event and select the relevant track from the track lists.

Each media track is identified by the following parameters:

* <code>id</code> : typically mapped to the format used in the media container,
* <code>kind</code> : particularly relevant are 'main' and 'captions',
* <code>label</code> : to be presented to the user,
* <code>language</code>.

A single video track can be active at a given time: the currently active video track can be set programmatically using the <code>selected</code> attribute:

{% highlight javascript %}
    video.onloadedmetadata = function (e) {
        for(i=0;i<video.videoTracks.length;i++) {
            if (video.videoTracks[i].kind == 'alternative') {
                video.videoTracks[i].selected = true;
            }
        }
    } 
{% endhighlight %}

Alternatively, it can be selected declaratively using a Media fragment URI in the form <code>track=label</code>:

{% highlight html %}
    <video src="myvideo#track=Alternative"></video> 
{% endhighlight %}

Multiple audio tracks can be active at the same time: in that case, their audio will be mixed.

A specific audio track can be selected programmatically by setting its <code>enabled</code> attribute to true.

The example below illustrates how a single audio language can be selected programmatically:

{% highlight javascript %}
    video.onloadedmetadata = function (e) {
        for(i=0;i<video.audioTracks.length;i++) {
            if ((video.audioTracks[i].kind == 'main') 
             && (video.audioTracks[i].language == 'fr')) {
                var current = video.audioTracks.selectedIndex;
                video.audioTracks[current].enabled = false;
                video.audioTracks[i].enabled = true;
            }
        }
    } 
{% endhighlight %}

Inband media tracks are however today not supported by any desktop browser, and even worse, the corresponding bindings into WebKit and Gecko are yet to be implemented.

## In-band text tracks

Inband text tracks are exposed through the <code>textTracks</code> attribute of the media element, and would become available as soon as the Media player has reached the <code>HAVE_META_DATA</code> state.

Each Text track is composed of a list cues that represent individual pieces of timed metadata. If a text track is active (see below), the user-agent will generate an <code>cue</code> event every time it reaches a point in the timeline that corresponds to a cue.

A specific text track can have three different states that are controlled by its <code>mode</code> attribute:

* <code>disabled</code>: the track is simply ignored by the user-agent,
* <code>hidden</code>: the track is active, the user-agent generates events for the track cues, but nothing is displayed on screen,
* <code>showing</code>: the track is active, the user-agent generates events for the track cues, the cues are displayed on screen (if the track is a subtitle track).

The example below illustrates how to activate and display a french subtitles text track as an overlay to the video content:

{% highlight javascript %}
    for(var j=0; j < tracks.length;j++){
        track = video.textTracks[j];
        if(track.kind === "subtitles"){
            if(track.language == 'fr'){
                track.mode = 'showing';
            }else{
                track.mode = 'disabled';
            }
        }
    }
{% endhighlight %}

Inband text tracks are supported only by Safari on desktop.

# Out-of-band text tracks

In addition to in-band media tracks, the HTML5 Media element supports out-of-band text tracks that can be used to complement a media with subtitles, audio descriptions, chapters or any kind of metadata to be synchronized with the multimedia content. 

Out-of-band Text Tracks can be specified declaratively as children of a media element using the <code>track</code> element:

{% highlight html %}
    <video src="sintel.mp4">
      <track kind="subtitles" 
             label="English subtitles" 
             src="sintel_en.vtt" srclang="en" default></track>
      <track kind="subtitles"
             label="Sous-titres français"
             src="sintel_fr.vtt" srclang="fr"></track>
    </video>
{% endhighlight %}

Alternatively, they can be built entirely from javascript using the <code>addTextTrack</code> method. Please refer  to the [specification](http://www.w3.org/TR/html5/embedded-content-0.html#dom-media-addtexttrack) for details.

Once registered, the out-of-bands text tracks are available like any other test track through the <code>textTracks</code> attribute of the Media element. 

Out-of-band text tracks are supported by Safari and Google Chrome on desktop. Both will however only display subtitles using the [WebVTT](http://dev.w3.org/html5/webvtt/) format.

For a more detailed introduction to Out-of-band Text Tracks, please refer to this [article](http://www.html5rocks.com/en/tutorials/track/basics/).
  

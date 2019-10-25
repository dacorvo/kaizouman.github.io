---
layout: post
title: 'Subtitles and Chaptering using Timed Text Tracks'
date: '2013-03-27 19:00:00'
categories:
- Web Development
tags: HTML5 video subtitles chapters webvtt
published: true
redirect_from: /2013/03/subtitles-and-chaptering-using-text-tracks/
---
The HTML5 Media Elements primarily support audio and video, but media presentation can be enhanced using text tracks containing individual pieces of metadata distributed along the media timeline.

In this post, I will describe how these [Timed Text Tracks](http://www.w3.org/TR/html5/embedded-content-0.html#timed-text-tracks) can be used to add subtitles and chaptering to an existing video.

<!--more-->

## About Timed Text Tracks
As defined by the HTML5 specification, [Timed Text Tracks](http://www.w3.org/TR/html5/embedded-content-0.html#timed-text-tracks) are sequences of cues - small pieces of textual information - orderly distributed along a media timeline.

Text tracks can exist in different flavors:

* subtitles
* captions
* descriptions
* chapters
* metadata

The behaviour of the user-agent specific to the different kind of text tracks is not detailed in the HTML5 specification (although it seems pretty obvious what the browser should do when it loads a subtitle track).

The common behaviour of the user-agent is however detailed:

* the text-track shall be made available to the web application through a TextTrack object stored in the <code>textTrackList</code> attribute of the Media Element,
* whenever the media timeline encounters a cue, generate a <code>cuechange</code> event on the track and an <code>enter</code> event on the cue,
* whenever the media timeline goes beyond a cue, generate an <code>exit</code> event on the cue.

This is for instance how an application would subscribe to events generated when the timeline reaches the beginning of a chapter during playback:

    tracks = video.textTracks;
    for(var j=0; j < tracks.length;j++){
        track = video.textTracks[j];
        if(track.kind === "chapters"){
            track.addEventListener("cuechange",displayChapter,false);
        }
    }

Text tracks can be embedded in the media content ("inband" text tracks), or simply referenced as children of the Media Element ("out-of-band" text tracks).

When referencing an out-of-band text track, the browser will enforce cross-origin policy restrictions, so be prepared to use [CORS](http://www.w3.org/TR/cors/) if the track file is not hosted on the same domain as your web page.

Note: HTML5 rocks has published a nice [tutorial](http://www.html5rocks.com/en/tutorials/track/basics/) on out-of-band text tracks.

## Adding Subtitles
Unlike content produced for TV, most of the media available today on the internet doesn't include inband subtitles: most of the time the closed captions are available as a separate file per language.

Being able to add these subtitles by just referencing them is therefore a very appealing feature, but there is, as always, a caveat ...

The thing is that Web people had an attack of [NIH syndrom](http://en.wikipedia.org/wiki/Not_invented_here) when they added the Time Text Tracks support to the browser, and although any format could theoretically be supported, they nevertheless decided to reinvent a new format named [WebVTT](http://dev.w3.org/html5/webvtt/) for describing text tracks, and guess what, it is today the only one that is actually supported ...

To be fair, WebVTT, formerly known as WebSRT, is a slightly modified version of the popular SubRip format (.srt).

The main differences between SubRip and WebVTT are:

* WebVTT's first line starts with WEBVTT
* All characters are UTF-8
* CSS is used instead of the FONT tag (that actually makes sense)

Anyway, it is not that difficult to convert an existing .srt file to a WebVTT file. There is even an [online converter](http://atelier.u-sub.net/srt2vtt/) provided by [delphiki](https://twitter.com/delphiki).

Note: You could even do it on the fly in javascript by parsing the SRT and re-creating the cues.

A WebVTT subtitle file looks like this:

    WEBVTT
    
    1
    00:01:47.250 --> 00:01:50.500
    This blade has a dark past.

    2
    00:01:51.800 --> 00:01:55.800
    It has shed much innocent blood.
    
    3
    00:01:58.000 --> 00:02:01.450
    You're a fool for traveling alone,
    so completely unprepared.

Once you have a working WebVTT subtitle file, all you need to do is to reference it as a child of your Media Element with <code>kind = subtitles</code>:

    <video src="sintel.mp4">
      <track kind="subtitles" 
             label="English subtitles" 
             src="sintel_en.vtt" srclang="en" default></track>
      <track kind="subtitles"
             label="Sous-titres français"
             src="sintel_fr.vtt" srclang="fr"></track>
    </video>

If you are using native controls in your browser, you should see an icon allowing you to activate subtitles (the browser will automatically pick the subtitles track defined as default, or the first one it encounters), and possibly even a list of subtitles to choose from (unfortunately not at the time this article is written).

If you want to use your own controls, you can activate subtitles programmatically:

    tracks = video.textTracks;
    for(var j=0; j < tracks.length;j++){
        track = tracks[j];
        if(track.kind === "subtitles"){
            if(track.language == 'fr'){
                track.mode = 'showing';
            }else{
                track.mode = 'disabled';
            }
        }
    }

## Adding Chapters
WebVTT can be used to describe subtitles, but also chapters. 

The syntax is pretty much the same as the one used for subtitles:

    WEBVTT
    
    00:00:00.000 --> 00:01:45.999
    Opening credits
    
    00:01:46.000 --> 00:02:37.999
    A dangerous quest
    
    00:02:38.000 --> 00:04:46.999
    Scales

Again, adding support for chaptering to a Media content is as easy as referencing the corresponding WebVTT file with <code>kind = chapter</code>:

    <video src="sintel.mp4">
      <track kind="chapter" 
             label="Chapters" 
             src="sintel_chapters.vtt" srclang="en"></track>
    </video>

It is not entirely clear yet what the native interface would be when presenting a content described in chapters, and it is not supported yet anyway, so the best option is to set the corresponding track to hidden, which tells the browser to process the text track without displaying it, and provide our own user interface for chaptering:

Typically, we would like to provide the following features:

* to list chapters,
* to allow the user to select a chapter,
* to highlight the current chapter during playback.

We will attach the code to hide the track and list chapters to the <code>load</code> event of the <code><track></code> element:

    <video src="sintel.mp4">
      <track kind="chapter" 
             label="Chapters" 
             src="sintel_chapters.vtt" srclang="en"
             onload = "displayChapters(this)"></track>
    </video>

    function displayChapters(trackElt){
        if((trackElt) && (textTrack = trackElt.track)){
            if(textTrack.kind === "chapters"){
                // Do not show the track
                textTrack.mode = 'hidden';
                var chapterBlock = document.getElementById("chapters");
                // List cues
                for (var i = 0; i < textTrack.cues.length; ++i) {
                    var cue = textTrack.cues[i];
                    var chapterName = cue.text;
                    var start = cue.startTime;
                    // Design an interface here allowing to choose a chapter
                    ....
                }
            }
        }
    }

To select a chapter we tell the browser to seek to the corresponding position in the timeline:

    video.currentTime = cue.startTime 

To highlight the current chapter, we listen to the <code>cuechange</code> event:

    function displayChapters(trackElt){
        if((trackElt) && (textTrack = trackElt.track)){
            ....
            textTrack.addEventListener("cuechange",
                                        function(evt){
                                            // Assuming here there is only one cue active, ie
                                            // chapters are not overlapping
                                            var cue = this.activeCues[0];
                                            // Use cue parameters to identify chapter and
                                            // provide visual feedback
                                            ...
                                        },
                                        false);
        }
    }

And that's about it !

## Demo

Click on the image below to see how it works when all the pieces are put together:

<div class='picture'>
<a href='/demos/video-text-track-demo/'>
<img src='/demos/video-text-track-demo/demo.png'>
</a>
</div>

## Browser support

At the time this article is written, out-of-band Timed Text Tracks using the WebVTT format are supported by Chrome and Internet Explorer 10.

---
layout: post
title: 'Why Google Dropped WebKit'
date: '2013-04-04 19:00:00'
categories:
- Web Browsers
tags: WebKit Blink Google
published: true
redirect_from: /2013/04/why-google-dropped-webkit/
---
Google announced yesterday in a [short post](http://blog.chromium.org/2013/04/blink-rendering-engine-for-chromium.html) on the Chromium Blog that they were dropping WebKit for Blink, a new rendering engine.
 
I was at first taken aback by the announcement, but after a while, I realized that I should have seen it coming ...  

I would summarize their decision in three words: Rationalization, Emancipation, and Differentiation.

<!--more-->

I have been working with WebKit for quite a while now, mainly integrating the smaller ports (Qt, GTK, EFL) on embedded targets for my employer. 

One of my main concern has always been to make sure that the port I was basing a product on was not a dead-end, ie that there was a big enough community backing it to make sure that the continuous stream of features got integrated in a timely fashion.

After Nokia dropped Qt I was wondering what impact it would have on QtWeKit, so I started to monitor closely who was doing what on WebKit.

I extracted in particular [a large amount of information](/tools/webkit-who) from the commit logs to try to find out if the Qt port could survive without Nokia.

I still haven't made up my mind, although there's been a serious drop of contributions on the Qt port lately … but that's another story.

I was anyway probably too focussed on QtWebKit though, as I didn't see the obvious ... 

## Drop the burden  

Looking at the WebKit commit logs, the first things that strikes you is that Google (and not Apple) is by far the first contributor to WebKit in terms of volume.

This has been heavily discussed in the past, when [Evan Martin](http://neugierig.org/software/chromium/notes/2010/02/webkit-commits.html) posted a graph showing that Google contributions had started exceeding Apple's around the end of 2009. The conclusion at that time was that Apple was still the main contributor to WebKit because they contributed more to the core and had more reviewers.

From the statistics I extracted recently, things have changed.

First, Google alone contributes to half of the commits made on WebKit, whereas Apple weights only a quarter.

<div class='picture'>
<a href='/tools/webkit-who/'>
<img src='/images/posts/webkit-google-apple-2008-2013.png'></img>
</a>
<div class='legend'>Google vs Apple WebKit commits 2008-2013</div>
</div>

Second, Google contributes equally to the core and the chromium port.

<div class='picture'>
<a href='/tools/webkit-who/'>
<img src='/images/posts/webkit-google-2012-2013.png'></img>
</a>
<div class='legend'>Google WebKit commits by type 2012-2013</div>
</div>

Third, Google is the main contributor on maintenance tasks, including in particular the WebKit 'gardening', ie the test regressions tracking.

<div class='picture'>
<a href='/tools/webkit-who/'>
<img src='/images/posts/webkit-google-maintenance-2012-2013.png'></img>
</a>
<div class='legend'>Google WebKit Maintenance commits 2012-2013</div>
</div>

The latter is probably the most interesting: since 2009, the number of maintenance commits has tripled, following the overall inflation of revisions. Google being the main committer used to take most of the maintenance burden on its (large) shoulders, and I can imagine they decided it was enough ... 

This is probably what Alan Barth means when he says: "Over the long term a healthier codebase leads to more stability and fewer bugs".

## Emancipate

Despite Google being the main driving force, WebKit is still Apple's baby. 

If you take a look at the WebKit Team page, you realize that Google has 95 members, but only 36 reviewers, whereas 48 out of 59 Apple contributors are reviewers. 

More important, the main design choices are still under Apple's control, and they don't necessarily agree with Google: see for instance the different views on multi-process architecture also mentioned by Adam Barth (as a matter of fact, WebKit2 has introduced a multi-process architecture, but different from the one used in Chrome).

At some point, Adam's team mates probably decided that it was time for them to kill the father …

## Differentiate from the competitors

This is probably something they won't easily admit, but WebKit success on the embedded market may have been an issue for Google: if any device manufacturer is able to reproduce the 'WebKit experience' on their devices, it becomes more difficult to nourish the momentum around Android.

Switching to Blink may thus also be a tactical move to allow them to differentiate themselves from their competitors: although Blink will be Open Source, Google seems to take for granted that only a limited set of build configurations will be supported (including obviously Android). I therefore doubt it will be ever ported to a 'base' embedded distribution.

But I hope they will prove me wrong …


 

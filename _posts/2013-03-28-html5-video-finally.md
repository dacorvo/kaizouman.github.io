---
layout: post
title: 'HTML5 Video, finally'
author: 'David Corvoysier'
categories: TV
tags: HTML5 video EME DRM
published: true:
redirect_from: /2013/03/html5-video-finally/
---
It's been quite a while since HTML5 video was introduced, and Adobe and Microsoft have stopped promoting Flash and Silverlight since early 2011, but it is only since a few days that the first HTML5 based commercial video service has been [launched](http://www.webpronews.com/samsung-chromebook-gets-html5-netflix-streaming-2013-03).

Why did it take so long ? Well, there were surely a few hurdles to overcome, but only one was really a show-stopper ...

<!--more-->

## Elephant in the room: any commercial video service requires DRM 

One may disagree with Digital Rights Management for many legitimate reasons, and I will not try to make a point for or against here: the thing is that today, it is mandated by the film industry to implement some kind of DRM to distribute their content. And that's about it.
 
Some people may argue that it will change in the future, taking the music industry as an example. They may be right ... eventually, but for now, DRM cannot be avoided.

## A tactical move from Netflix, Microsoft and Google

Netflix has been a supporter of HTML5 video for a very long time, and one of the first companies to bring requirements to the W3C to allow the deployment of Commercial Video Services using standard Web technologies.

They joined together with Microsoft, Opera, Comcast and several TV service providers (including Orange) into the W3C Web & TV interest group and started pushing requirements to the HTML Working Group for inclusion into HTML5. The reactions were however not up to their expectations, most of the corresponding bugs being rejected, sometimes with [ironic comments](https://www.w3.org/Bugs/Public/show_bug.cgi?id=10904#c4) from the HTML editor.

This was [particularly true for DRM support](https://www.w3.org/Bugs/Public/show_bug.cgi?id=10902#c8), which led Netflix and Microsoft to realize that they'd better implement something by themselves if they wanted to get things done. 

And that's just what they did, teaming with Google to produce the [Encrypted Media Extensions](https://dvcs.w3.org/hg/html-media/raw-file/tip/encrypted-media/encrypted-media.html) specification.

## W3C blessing, but fierce opposition from the community

The first time the idea of a W3C specification to add DRM support into HTML was [submitted](http://lists.w3.org/Archives/Public/public-html/2012Feb/0273.html) to the HTML Working Group about a year ago, it immediately raised a huge controversy, and infuriated messages were exchanged on the mailing-list.

Note: For those interested, there is a wiki page detailing the [pros & cons](http://www.w3.org/community/pua/wiki/Digital_Rights_Management) of DRM in the browser.

The W3C managed to cool things down for a while by moving the discussion about DRM in a dedicated "HTML Media task force", also in charge of producing a specification targeting adaptive streaming in the browser.

The controversy restarted with an equal intensity however when a Call for Consensus (CfC) was [issued](http://lists.w3.org/Archives/Public/public-html-admin/2013Jan/0102.html) to promote the EME specification to the status of "First Public Working Draft", which would have meant that it was officially endorsed by the HTML Working Group.

Despite [a formal statement by the W3C](http://lists.w3.org/Archives/Public/public-html-admin/2013Feb/0122.html) that DRM was "in scope" for the Web platform, the CfC was rejected, at least temporarily. 

But it didn't matter: the move had been made.

## Business prevails 

The EME may well has been denied the right to move to draft standard, it didn't prevent Google from [implementing EME on Google Chrome Books](https://plus.google.com/+chrome/posts/GNsoSDLRRY6), and Netflix from launching their video service on these devices.

It shouldn't take long for Microsoft to follow with an EME implementation on their devices, and looking at the list of [EME supporters](http://www.w3.org/community/pua/wiki/Digital_Rights_Management#Businesses_supporting_the_EME_FPWD_CfC), we should see more EME-backed video services launched by the end of the year. 

Actually, it doesn't matter anymore if EME is endorsed or not by the W3C: the most important thing was to bring the relevant people together to build the specification. Now it has been done, its success in not in the hand of W3C anymore, but in those of implementers.

Because in the end, the one who wins is the one who delivers.   

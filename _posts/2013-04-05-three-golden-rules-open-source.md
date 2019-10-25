---
layout: post
title: 'Three golden rules of working with Open Source Software'
date: '2013-04-05 17:00:00'
categories:
- Web Development
tags:
- Open Source
type: post
published: true
redirect_from: /2013/04/three-golden-rules-open-source/
---
Due to the increasing complexity of computer systems, you cannot develop a product from scratch anymore (well, actually you could, but it would take so much time that it would not be worth the benefits you may get from it). So either you have a huge legacy codebase that you can rely on (and your name is Microsoft), or you have to build something on top of Open Source software.

I have been working with Open Source software for several years, and I have witnessed striking successes and miserable failures, up to the point that I started to see a pattern emerging.
This is what I want to share with you.

<!--more-->

## Do not fork
Starting a new product based on Open Source software is often very exciting at the beginning, because you usually manage to have something usable in a very short timeframe (at least as compared to the time it would have taken to develop everything from scratch).

After a while though, comes the disappointment: unless you are very lucky, it is unlikely that the software you integrated fully covers your needs. 

The problem is, Open Source software is no different than proprietary software: you only get what you paid for, and since you paid nothing, it is up to you to add what is missing.

Every software component has a learning curve, and Open Source components make no exception: what makes things often more difficult is the lack of documentation and the availability of people with sufficient knowledge to guide you.

That's where the most dangerous temptation comes: why bother with foreign design constraints and coding rules when you could take shortcuts and deliver your project faster ? The danger is even greater in companies with a strong development culture that may be very tempted to substitute their standards to the ones used in the open Source projects they are basing their work on. 

The trouble is, when you have decided to fork an Open Source project, there is no turning back: from that moment on, you're on your own.

Forking is the main cause of failure for projects based on Open Source: over time, both the original project and its fork advance in parallel, up to the point where it becomes impossible to merge even simple bug fixes. This is the time of tough decisions: basically you have to choose between loosing all you've invested in your project or staying apart from the main stream and its benefits. 
  
So my first advice is: do not fork.

## Push your changes back upstream
Most companies are very reluctant to develop in the Open: that's a cultural thing. 

To be fair, even if the licence of most Open Source components will force you to publish your changes when you release a product, nothing prevents you from keeping them private during the development phase. 

So, the first thing you will be asked when you start a new project based on Open Source software is often to replicate everything in a private corporate environment. You can even be asked to release your changes at the end of the project in a separate public repository to limit the chances some competitor might use them (nasty !).

Beware though, as too much privacy might also bring the seeds of doom to your project.

A few words on contributing back changes, first: this undoubtedly requires an extra effort, and you may sometimes be facing some resistance from the project reviewers. This is however worth it. Big times.

One thing Open Source newbies often ignore is that once a change has been accepted upstream, it cannot be undone without a strong consensus inside the project. Even better, contributors bringing new changes to the project are responsible for preserving it: you cannot make a change that breaks things (at least as long there is someone left to complain). It is actually not unusual to see other contributors adapting code you have contributed as the project evolves. 

In the contrary, nothing prevents someone to post changes to the project that breaks stuff you haven't submitted (yet). This is actually very likely to happen as time advances: the more you wait to submit your changes, the most difficult it will be to merge them back upstream. 

My second advice is therefore: push you changes back upstream as soon as possible. 

## Keep your development tree sync'ed
Active Open Source projects have a very fast pace of innovation, up to the point that incompatible API or ABI changes may be submitted within the typical timeframe of a project. 

Even worse, popular Open Source components are so intricated that incompatible changes on one project can propagate to the others. 

It is therefore very important to keep your development tree synchronized with recent upstream revisions to avoid being pushed to a dead branch without even knowing it. 

My prediction is that in less than five years from now, it won't be possible to develop anything based on Open Source without being synchronized on a daily basis with upstream repositories.

My final advice it thus to keep your tree sync'ed on a regular basis (typically weekly or bi-weekly).

## To summarize
These are my three simple advices if you want to succeed in delivering products based on Open Source components:

* Do Not Fork,
* Push back your changes upstream,
* Keep updated on a regular basis. 

There is however an even better advice that embraces them all: why don't you develop in the Open ?


---
layout: post
title: Generating custom javascript events
categories:
- Web Development
tags: Custom Event Javascript
published: true
redirect_from: /2010/03/generating-custom-javascript-events/
---
HTML 5 introduces many new exciting features, and it is very tempting 
for the web developer to start buiding pages taking advantage of this 
new syntax as soon as an early implementation shows up.

<!--more-->

For those aiming at the masses, it may however be necessary to fill the 
HTML 5 implementation gaps of some browsers with custom javascript 
implementations.

Fair enough, says the web developer: thanks to Javascript extensibility 
mechanisms, it is easy to add new properties and methods to existing 
objects and even to create new javascript objects, but what about all 
these new javascript events ?

Well, since DOM level 2 specification it is also possible to 
[create](http://www.w3.org/TR/DOM-Level-2-Events/events.html#Events-document),
[initialize](http://www.w3.org/TR/DOM-Level-2-Events/events.html#Events-interface) and
[dispatch](http://www.w3.org/TR/DOM-Level-2-Events/events.html#Events-Registration-interfaces)
custom javascript events.

Basically you can create a javascript event by calling the DOM document <code>createEvent</code> method:

    
    var evt = document.createEvent("Event");
    
    ...
    
    var mevt = document.createEvent("MouseEvent");
    

The <code>createEvent</code> argument defines the <a href="http://www.w3.org/TR/DOM-Level-2-Events/events.html#Events-eventgroupings-uievents" alt="1.6.1. User Interface event types">Event interface</a>:

* <code>UIEvents</code> : for DOMFocusIn, DOMFocusOut & DOMActivate events,
* <code>MouseEvents</code>: for click, mousedown/up/over/move/out,
* <code>MutationEvents</code>: for all DOMXX modification events,
* <code>HTMLEvents</code>: for load, unload, abort, error, select, change, submit, reset, focus, blur, resize, scroll events,
* <code>Event</code>: Any other event.

Note: You cannot generate a key event

Once you've instantiated an event of a specific type, you can specify 
its attributes using the corresponding <code>initEvent</code> method:

    var evt = document.createEvent("Event");
    
    evt.initEvent("myEvent",true,true); 
    
    ...
    
    var mevt = document.createEvent("MouseEvent");
    
    mevt.initMouseEvent("click",true,true,...); 

The first three parameters are common to all initEvents functions: 
the first argument specifies the event type, the second indicates if the
 event can bubble and the third if its default action can be prevented.

Once the event has been created and initalized, you can further customize
 it by adding specific attributes you may want to pass to the listening function:

    evt.foo = "bar";

Once done, the event can be dispatched to the document or to a specific node:
    
    document.dispatchEvent(evt);

or

    var target = document.getElementById("myTarget");
    
    target.dispatchEvent(evt);

Of course, you also need to specify an event handler if you actually 
want to catch this new event:

    document.addEventListener("myEvent",myEventHandler,false);

or

    var target = document.getElementById("myTarget");
    
    target.addEventListener("myEvent",myEventHandler,false);

<script type="text/javascript">
function customJSEvent(){
var evt = document.createEvent("Event");
evt.initEvent("myEvent",true,true); 
evt.foo = "bar";
document.dispatchEvent(evt);
}
function handleEvent(e){
alert("OK: Received "+ e.type + " event with foo=" + e.foo);
}
document.addEventListener("myEvent",handleEvent,false);
</script>

Does you browser support custom JS events ?

<a href="#" onclick="customJSEvent();">Click</a> to check.

Update (2013): This method of creating events is deprecated in favor of the newest [DOM4 Custom Events](https://developer.mozilla.org/en-US/docs/Web/API/CustomEvent)

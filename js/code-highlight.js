(function(){
/* Look for <pre> elements in the code */
var code = false;
var preblocks = document.getElementsByTagName('pre');
for (var i=0; i< preblocks.length; i++) {
    if (preblocks[i].className.indexOf('prettyprint') != -1) {
        code = true;
    } else if (preblocks[i].getElementsByTagName('code').length > 0) {
        preblocks[i].className+= ' prettyprint';
        code = true;
    }
}
if (code) {
    var link = document.createElement('link');
    link.rel = 'stylesheet';
    link.media = 'screen';
    link['data-noprefix'] = true;
    link.href = 'https://raw.githubusercontent.com/googlearchive/code-prettify/master/styles/sunburst.css';
    document.head.appendChild(link);
    var script = document.createElement('script');
    script.type = 'text/javascript';
    script.src="https://cdn.jsdelivr.net/gh/google/code-prettify@master/loader/run_prettify.js";
    document.head.appendChild(script);
}
})();

(function(_ds){var window=this;'use strict';var g1=function(a){return(0,_ds.T)('<span class="devsite-tooltip-msg">'+_ds.V(a.Rq)+"</span>")},i1=function(){var a=_ds.rE.call(this)||this;a.j=new _ds.zl;a.m=null;a.o=null;a.C=new _ds.Dn(function(){return h1(a)},250);return a},m1=function(a){a.j.listen(document.body,"devsite-content-updated",function(){return a.C.xa()});a.j.listen(document.body,"onpointermove"in window?"pointermove":"mousemove",function(b){b=b.target;for(var c=!1;b;){b.hasAttribute&&(c=j1(b));if(c)break;b=b.parentNode}b&&
c?k1(a,b):l1(a)});a.j.listen(document.body,"focusin",function(b){b=b.target;var c=b.firstElementChild;(b.classList.contains("devsite-nav-title")&&c?j1(c):j1(b))?k1(a,b):l1(a)});a.j.listen(document.body,["devsite-sticky-scroll","devsite-sticky-resize"],function(){return l1(a)})},j1=function(a){return a.hasAttribute("no-tooltip")?!1:a.hasAttribute("data-title")||a.hasAttribute("data-tooltip")||a.hasAttribute("tooltip")&&a.clientWidth<a.scrollWidth},k1=function(a,b){if(a.m!==b){l1(a);var c=b.getAttribute("data-tooltip")||
b.getAttribute("data-title")||b.textContent.trim(),d=_ds.ol(g1,{Rq:c});d.style.opacity=0;document.body.appendChild(d);var e=_ds.pk(window),g=_ds.Un(b),h=_ds.Un(d),k=_ds.Pn(b);c=k.y+g.height;c+h.height+8>e.height&&(c=k.y-h.height-16);g=k.x+g.width/2-h.width/2;h.width>e.width?g=0:(g=Math.max(g,8),e=e.width-(g+h.width+8),0>e&&(g=g+e-8));d.style.top=c+"px";d.style.left=g+"px";a.m=b;a.o=d;window.requestAnimationFrame(function(){d.style.opacity=1})}},l1=function(a){if(a.m){a.m=null;var b=a.o;a.o=null;_ds.oh(b,
_ds.Wl,function(){_ds.Fk(b);_ds.Bk(b)});window.setTimeout(function(){_ds.Fk(b);_ds.Bk(b)},1E3);b.style.opacity=0}},h1=function(a){Array.from(document.querySelectorAll(".devsite-article-body [title]")).forEach(function(b){b.setAttribute("data-title",b.getAttribute("title"));b.removeAttribute("title")});a.hasAttribute("blocked-link")&&Array.from(document.getElementsByTagName("a")).forEach(function(b){if(b.hasAttribute("href")){for(var c=(new URL(b.getAttribute("href"),document.location.origin)).hostname.replace("www.",
""),d=0;d<n1.length;d++)if(-1!==c.indexOf(n1[d]))return;for(d=0;d<o1.length;d++)if(-1!==c.indexOf(o1[d])){b.setAttribute("data-title","This link may not be accessible in your region.");b.removeAttribute("title");break}}})};var n1=["dl.google.com"],o1="abc.xyz admob.com android.com blogger.com blogspot.com chrome.com chromium.org domains.google doubleclick.com feedburner.com g.co ggpht.com gmail.com gmodules.com goo.gl google.com google.org googleapis.com googleapps.com googlecode.com googledrive.com googlemail.com googlesource.com googlesyndication.com googletagmanager.com googleusercontent.com gv.com keyhole.com madewithcode.com panoramio.com urchin.com withgoogle.com youtu.be youtube.com ytimg.com".split(" ");
_ds.v(i1,_ds.rE);i1.prototype.connectedCallback=function(){document.body.hasAttribute("touch")?_ds.Fk(this):(m1(this),this.C.xa())};i1.prototype.disconnectedCallback=function(){_ds.rE.prototype.disconnectedCallback.call(this);_ds.Fl(this.j)};i1.prototype.disconnectedCallback=i1.prototype.disconnectedCallback;i1.prototype.connectedCallback=i1.prototype.connectedCallback;try{window.customElements.define("devsite-tooltip",i1)}catch(a){console.warn("devsite.app.customElement.DevsiteTooltip",a)};})(_ds_www);
(function(_ds){var window=this;'use strict';var WY=function(a){var b=arguments.length;if(1==b&&Array.isArray(arguments[0]))return WY.apply(null,arguments[0]);for(var c={},d=0;d<b;d++)c[arguments[d]]=!0;return c},XY=function(a,b,c,d,e,g,h,k,l,m){this.h=a;this.C=b;this.G=c;this.m=d;this.o=e;this.D=g;this.F=h;this.O=k;this.j=l;this.J=m},YY=function(a){return new XY(a.m,a.o,a.O,a.h,a.j,a.C,a.J,a.F,a.D,a.G)},ZY=function(a,b){a.G=b;return a},$Y=function(a,b){a.D=b;return a},aZ=function(a,b){a.F=b;return a},bZ=function(){this.m=null;
this.o="";this.G=this.D=this.F=this.J=this.C=this.j=this.h=this.O=null},cZ=function(a){var b=a.ma;b=(b=b&&"composed"in b&&b&&"composedPath"in b&&b.composed&&b.composedPath())&&0<b.length?b[0]:a.target;return YY(ZY($Y(aZ((new bZ).keyCode(a.keyCode||0).key(a.key||"").shiftKey(!!a.shiftKey).altKey(!!a.altKey).ctrlKey(!!a.ctrlKey).metaKey(!!a.metaKey).target(a.target),b),function(){return a.preventDefault()}),function(){return a.stopPropagation()}))},dZ=function(a,b,c){_ds.bh.call(this,a,c);this.j=b},
eZ=function(a){this.h=a||null;this.next=a?null:{}},fZ=function(a,b){b.shift().forEach(function(c){var d=a[c];d&&(0==b.length?d.h&&delete a[c]:d.next&&(fZ(d.next,b.slice(0)),_ds.$b(d.next)&&delete a[c]))})},iZ=function(a){a=a.replace(/[ +]*\+[ +]*/g,"+").replace(/[ ]+/g," ").toLowerCase();a=a.split(" ");for(var b=[],c,d=0;c=a[d];d++){var e=c.split("+"),g=null,h=null;c=0;for(var k,l=0;k=e[l];l++){switch(k){case "shift":c|=1;continue;case "ctrl":c|=2;continue;case "alt":c|=4;continue;case "meta":c|=
8;continue}e=void 0;g=k;if(!gZ){h={};for(e in hZ)h[hZ[e]]=_ds.EB(parseInt(e,10));gZ=h}h=gZ[g];g=k;break}b.push({key:g,keyCode:h,modifiers:c})}return b},jZ=function(a,b,c){c=c||0;b=["c_"+b+"_"+c];""!=a&&b.push("n_"+a+"_"+c);return b},kZ=function(a,b){if("string"===typeof b[a])a=iZ(b[a]).map(function(e){return jZ(e.key||"",e.keyCode,e.modifiers)});else{var c=b,d=a;Array.isArray(b[a])&&(c=b[a],d=0);for(a=[];d<c.length;d+=2)a.push(jZ("",c[d],c[d+1]))}return a},lZ=function(a,b){32==a.D&&32==b.h&&(0,b.j)();
a.D=null},mZ=function(a){return _ds.qJ&&a.o&&a.m},nZ=function(a,b,c){var d=b.shift();d.forEach(function(e){if((e=a[e])&&(0==b.length||e.h))throw Error("Keyboard shortcut conflicts with existing shortcut: "+e.h);});b.length?d.forEach(function(e){e=e.toString();var g=new eZ;e=e in a?a[e]:a[e]=g;nZ(e.next,b.slice(0),c)}):d.forEach(function(e){a[e]=new eZ(c)})},oZ=function(a,b){for(var c=0;c<b.length;c++){var d=a[b[c]];if(d)return d}},pZ=function(a,b,c){a:{var d=b.h;if(""!=b.C){var e=b.C;if("Control"==
e||"Shift"==e||"Meta"==e||"AltGraph"==e){d=!1;break a}}else if(16==d||17==d||18==d){d=!1;break a}e=b.O;var g="TEXTAREA"==e.tagName||"INPUT"==e.tagName||"BUTTON"==e.tagName||"SELECT"==e.tagName,h=!g&&(e.isContentEditable||e.ownerDocument&&"on"==e.ownerDocument.designMode);d=!g&&!h||a.K[d]||a.F?!0:h?!1:a.L&&(b.m||b.o||b.D)?!0:"INPUT"==e.tagName&&a.R[e.type]?13==d:"INPUT"==e.tagName||"BUTTON"==e.tagName?a.N?!0:32!=d:!1}if(d)if(!c&&mZ(b))a.o=!1;else{c=_ds.EB(b.h);d=jZ(b.C,c,(b.G?1:0)|(b.o?2:0)|(b.m?4:
0)|(b.D?8:0));e=oZ(a.m,d);if(!e||1500<=Date.now()-a.C)a.m=a.j,a.C=Date.now();(e=oZ(a.m,d))&&e.next&&(a.m=e.next,a.C=Date.now());e&&(e.next?(0,b.j)():(a.m=a.j,a.C=Date.now(),a.G&&(0,b.j)(),a.J&&(0,b.J)(),d=e.h,e=a.dispatchEvent(new dZ("shortcut",d,b.F)),(e&=a.dispatchEvent(new dZ("shortcut_"+d,d,b.F)))||(0,b.j)(),_ds.Ln&&(a.D=c)))}},qZ=function(a,b,c){for(;0<c.length&&b;){var d=c.shift();if((d=oZ(b,d))&&(0==c.length&&d.h||qZ(a,d.next,c.slice(0))))return!0}return!1},tZ=function(a){_ds.Tm.call(this);
this.m=this.j={};this.C=0;this.K=WY(rZ);this.R=WY(sZ);this.G=!0;this.F=this.J=!1;this.L=!0;this.N=!1;this.D=null;this.h=a;_ds.ph(this.h,"keydown",this.wj,void 0,this);_ds.ph(this.h,"synthetic-keydown",this.Bj,void 0,this);_ds.qJ&&(_ds.ph(this.h,"keypress",this.Dj,void 0,this),_ds.ph(this.h,"synthetic-keypress",this.Ej,void 0,this));_ds.ph(this.h,"keyup",this.xj,void 0,this);_ds.ph(this.h,"synthetic-keyup",this.Cj,void 0,this)},uZ=function(a,b){return _ds.qg(a,10,b)},vZ=function(a){return(0,_ds.T)('<div class="devsite-popout" id="'+
_ds.W(a.id)+'"><div class="devsite-popout-result devsite-suggest-results-container" devsite-hide></div></div>')},wZ=function(a){var b='<button type="submit" class="button button-white devsite-search-project-scope">';a="All results in "+_ds.V(a);return(0,_ds.T)(b+a+"</button>")},xZ=function(a,b,c){var d='<button type="submit" class="button button-white devsite-suggest-all-results">';b?(a="All results across "+_ds.V(c),d+=a):(a='All results for "'+_ds.V(a)+'"',d+=a);return(0,_ds.T)(d+"</button>")},
yZ=function(a,b,c,d,e,g,h,k){a=(c?"":'<devsite-analytics-scope action="'+_ds.W(_ds.Nj("Restricted "+d))+'">')+'<a class="devsite-result-item-link" href="'+_ds.W(_ds.Nj(a))+'"><span class="devsite-suggestion-fragment">'+_ds.V(b)+"</span>"+(k?'<span class="devsite-suggestion-fragment">'+_ds.V(k)+"</span>":"")+(e?'<span class="devsite-suggestion-fragment">'+_ds.V(e)+"</span>":"")+(_ds.nj(g)&&!h?'<span class="devsite-suggestion-fragment">'+_ds.V(g)+"</span>":"");c||(a+='<span class="devsite-result-item-confidential">Confidential</span>');
return(0,_ds.T)(a+("</a>"+(c?"":"</devsite-analytics-scope>")))},zZ=function(a){var b=a.projectName,c=a.xk,d=a.query,e=a.Kk;a=a.hf;b='<div class="devsite-suggest-wrapper '+(e?"":"devsite-search-disabled")+'"><div class="devsite-suggest-section"><div class="devsite-result-label">There are no suggestions for your query</div></div>'+((e?'<div class="devsite-suggest-footer">'+(c?wZ(b):"")+xZ(d,c,a)+"</div>":"")+"</div>");return(0,_ds.T)(b)},AZ=function(a){var b=a.Dp,c=a.Op,d=a.Wp,e=a.projectName,g=a.xk,
h=a.query,k=a.bq,l=a.Kk;a=a.hf;var m='<div class="devsite-suggest-wrapper '+(l?"":"devsite-search-disabled")+'" tabindex="0" role="list"><div class="devsite-suggest-section">';if(0<d.length){m=m+'<div class="devsite-suggest-sub-section" role="listitem"><div class="devsite-suggest-header" id="devsite-suggest-header-partial-query" role="heading" aria-level="2">Suggested searches'+((g?'<span class="devsite-suggest-project">'+_ds.V(e)+"</span>":"")+'</div><devsite-analytics-scope category="Site-Wide Custom Events" label="Search" role="list" aria-labelledby="devsite-suggest-header-partial-query" action="Query Suggestion Click">');
for(var p=d.length,q=0;q<p;q++){var t=d[q];m+='<div class="devsite-result-item devsite-nav-label" id="suggestion-partial-query-'+_ds.W(q)+'" role="listitem" index=":'+_ds.W(q)+'">'+yZ(t.Qa(),t.Ha(),_ds.og(t,14),"Query Suggestion Click")+"</div>"}m+="</devsite-analytics-scope></div>"}m+=0<d.length&&0<b.length?'<hr role="none">':"";if(0<b.length){m=m+'<div class="devsite-suggest-sub-section" role="listitem"><div class="devsite-suggest-header" id="devsite-suggest-header-product" role="heading" aria-level="2">Pages'+
((g?'<span class="devsite-suggest-project">'+_ds.V(e)+"</span>":"")+'</div><devsite-analytics-scope category="Site-Wide Custom Events" label="Search" role="list" aria-labelledby="devsite-suggest-header-product" action="Page Suggestion Click">');p=b.length;for(q=0;q<p;q++)t=b[q],m+='<div class="devsite-result-item devsite-nav-label" id="suggestion-product-'+_ds.W(q)+'" role="listitem" index=":'+_ds.W(q)+'">'+yZ(t.Qa(),t.Ha(),_ds.og(t,14),"Page Suggestion Click",void 0,_ds.O(t,4),g)+"</div>";m+="</devsite-analytics-scope></div>"}m+=
0<k.length&&0<b.length+d.length?'<hr role="none">':"";if(0<k.length){m=m+'<div class="devsite-suggest-sub-section" role="listitem"><div class="devsite-suggest-header" id="devsite-suggest-header-reference" role="heading" aria-level="2">Reference'+((g?'<span class="devsite-suggest-project">'+_ds.V(e)+"</span>":"")+'</div><devsite-analytics-scope category="Site-Wide Custom Events" label="Search" role="list" aria-labelledby="devsite-suggest-header-reference" action="Reference Suggestion Click">');p=k.length;
for(q=0;q<p;q++)t=k[q],m+='<div class="devsite-result-item devsite-nav-label" id="suggestion-reference-'+_ds.W(q)+'" role="listitem" index=":'+_ds.W(q)+'">'+yZ(t.Qa(),t.Ha(),_ds.og(t,14),"Reference Suggestion Click",_ds.O(t,3),_ds.O(t,4),g,_ds.G(t,10)[0])+"</div>";m+="</devsite-analytics-scope></div>"}m+=0<c.length&&0<b.length+d.length+k.length?'<hr role="none">':"";if(0<c.length){m+='<div class="devsite-suggest-sub-section" role="listitem"><div class="devsite-suggest-header" id="devsite-suggest-header-other-products" role="heading" aria-level="2"><span role="columnheader">Products</span></div><devsite-analytics-scope category="Site-Wide Custom Events" label="Search" role="list" aria-labelledby="devsite-suggest-header-other-products" action="Product Suggestion Click">';
b=c.length;for(d=0;d<b;d++)k=c[d],m+='<div class="devsite-result-item devsite-nav-label" id="suggestion-other-products-'+_ds.W(d)+'" role="listitem" index=":'+_ds.W(d)+'">'+yZ(k.Qa(),k.Ha(),_ds.og(k,14),"Product Suggestion Click")+"</div>";m+="</devsite-analytics-scope></div>"}m+="</div>"+(l?'<div class="devsite-suggest-footer">'+(g?wZ(e):"")+xZ(h,g,a)+"</div>":"")+"</div>";return(0,_ds.T)(m)},BZ=function(){var a=_ds.R.call(this)||this;a.V=!1;a.J={};a.K="";a.F=null;a.m=new _ds.zl;a.j=null;a.N=!1;
return a},HZ=function(a){a.j&&(a.m.listen(a.j,"suggest-service-search",function(b){a:{var c=a.D.querySelector(".highlight");if(c&&(c=c.querySelector(".devsite-result-item-link"))){c.click();break a}CZ(a,b.detail.originalEvent,!!a.F)}}),a.m.listen(a.j,"suggest-service-suggestions-received",function(b){return void DZ(a,b)}),a.m.listen(a.j,"suggest-service-focus",function(){EZ(a,"cloud-track-search-focus",null);a.N=!0;FZ(a,!0)}),a.m.listen(a.j,"suggest-service-blur",function(){FZ(a,!1)}),a.m.listen(a.j,
"suggest-service-input",function(){!a.J["Text Entered Into Search Bar"]&&a.j.query.trim()&&(a.dispatchEvent(new CustomEvent("devsite-analytics-observation",{detail:{category:"Site-Wide Custom Events",label:"Search",action:"Text Entered Into Search Bar"},bubbles:!0})),a.J["Text Entered Into Search Bar"]=!0);a.N&&(EZ(a,"cloud-track-search-input",null),a.N=!1)}),a.m.listen(a.j,"suggest-service-navigate",function(b){return void GZ(a,b)}),a.m.listen(document.body,"devsite-page-changed",function(){return a.J=
{}}),a.j.Pk("SLASH",191));a.o&&a.m.listen(a.o,"submit",function(b){CZ(a,b)});a.D&&a.m.listen(a.D,"click",function(b){var c=b.target,d=c.closest(".devsite-result-item-link");d&&(FZ(a,!1),d="suggestion: "+d.getAttribute("href"),EZ(a,"cloud-track-search-submit",d));c.classList.contains("devsite-search-project-scope")&&CZ(a,b,!0)});a.Y&&a.m.listen(a.Y,"click",function(){return void FZ(a,!0)});a.R&&a.m.listen(a.R,"click",function(){return void FZ(a,!1)})},CZ=function(a,b,c){c=void 0===c?!1:c;var d,e;_ds.cb(function(g){if(1==
g.h){b.preventDefault();b.stopPropagation();if(!a.hasAttribute("enable-search"))return g.H(0);a.J["Full Site Search"]||(a.dispatchEvent(new CustomEvent("devsite-analytics-observation",{detail:{category:"Site-Wide Custom Events",label:"Search",action:"Full Site Search"},bubbles:!0})),a.J["Full Site Search"]=!0);EZ(a,"cloud-track-search-submit",a.L?"contains: "+a.L:"no match");d=c&&a.F?_ds.lm(a.F):_ds.lm(a.o.getAttribute("action"));e=new _ds.Ii(d.href);_ds.Vi(e,"q",a.j.query);d.search=e.h.toString();
return _ds.x(g,DevsiteApp.fetchPage(d.href),3)}FZ(a,!1);_ds.y(g)})},GZ=function(a,b){var c=b.detail;b=a.C.querySelector(".highlight");var d,e=Array.from(a.C.querySelectorAll(".devsite-result-item")),g=[],h=-1;if(b){var k=_ds.gl(b,function(m){return m.classList.contains("devsite-suggest-section")});g=Array.from(k.querySelectorAll(".devsite-result-item"));k=_ds.Ok(b.parentNode.parentNode);var l=_ds.Pk(b.parentNode.parentNode);h=e.indexOf(b)}switch(c.keyCode){case 37:if(!k&&!l)return;b&&(c=b.getAttribute("index"),
l?(d=l.querySelector('[index="'+c+'"]'))||(d=_ds.Hb(Array.from(l.querySelectorAll("[index]")))):k&&((d=k.querySelector('[index="'+c+'"]'))||(d=_ds.Hb(Array.from(k.querySelectorAll("[index]"))))));break;case 39:if(!k&&!l)return;b&&(c=b.getAttribute("index"),k?(d=k.querySelector('[index="'+c+'"]'))||(d=_ds.Hb(Array.from(k.querySelectorAll("[index]")))):l&&((d=l.querySelector('[index="'+c+'"]'))||(d=_ds.Hb(Array.from(l.querySelectorAll("[index]"))))));break;case 38:b?(d=e[h-1])||(d=_ds.Hb(g)):d=_ds.Hb(e);
break;case 40:b?(d=e[h+1])||(d=g[0]):d=e[0]}b&&(b.classList.remove("highlight"),b.removeAttribute("aria-selected"));d&&(a.G.setAttribute("aria-activedescendant",d.id),d.setAttribute("aria-selected","true"),d.classList.add("highlight"),d.scrollIntoViewIfNeeded&&d.scrollIntoViewIfNeeded()||d.scrollIntoView())},EZ=function(a,b,c){a.dispatchEvent(new CustomEvent(b,{detail:{type:"search",name:b,position:"nav",metadata:{eventDetail:c}},bubbles:!0}))},FZ=function(a,b){if(a.V!==b){_ds.KB(a.j,b);if(a.V=b)a.setAttribute("search-active",
"");else{var c=a.D.querySelector(".highlight");c&&c.classList.remove("highlight");a.removeAttribute("search-active");a.setAttribute("aria-expanded","false");_ds.Bk(a.C)}a.hasAttribute("capture")||a.dispatchEvent(new CustomEvent("devsite-search-toggle",{detail:{active:b},bubbles:!0}))}},DZ=function(a,b){a.L=null;b=b.detail;var c=b.suggestions,d=b.query;if(a.j.query.toLowerCase().startsWith(d.toLowerCase()))if(c){var e=c.Bb();0<e.length&&(b=e.filter(function(t){return t.Ha().includes(d)}),0<b.length&&
(a.L=b[0].Ha()));b=e.filter(function(t){return 2===t.Pc()});c=e.filter(function(t){return 3===t.Pc()});var g=e.filter(function(t){return 4===t.Pc()}).slice(0,5),h=e.filter(function(t){return 1===t.Pc()});e=b.length+g.length+h.length;for(var k=_ds.r(g),l=k.next();!l.done;l=k.next())l=l.value,l.Vc(_ds.lm((a.F||"/s/results")+"/?q="+l.Ha()).toString());var m=d.split(IZ);c.forEach(function(t){return uZ(t,_ds.G(t,10).filter(function(u){return m.some(function(w){return u.includes(w)})}))});k=a.getAttribute("project-name")||
"";l=a.hasAttribute("project-scope");var p=a.hasAttribute("enable-search"),q=a.getAttribute("tenant-name")||"";b={Dp:b,projectName:k,xk:l,Op:h,Wp:g,query:d,bq:c,Kk:p,hf:q};0===e?_ds.nl(a.C,zZ,b):(_ds.nl(a.C,AZ,b),JZ(a,d));a.setAttribute("aria-expanded","true");a.C.removeAttribute("hidden")}else a.C.setAttribute("hidden",""),a.setAttribute("aria-expanded","false")},JZ=function(a,b){b=new RegExp("("+_ds.Pd(b)+")","ig");a=_ds.r(a.C.querySelectorAll(".devsite-suggestion-fragment"));for(var c=a.next();!c.done;c=
a.next()){c=c.value;var d=c.innerHTML;d=d.replace(b,"<b>$1</b>");d=_ds.qE(d);_ds.Gd(c,d)}},hZ={8:"backspace",9:"tab",13:"enter",16:"shift",17:"ctrl",18:"alt",19:"pause",20:"caps-lock",27:"esc",32:"space",33:"pg-up",34:"pg-down",35:"end",36:"home",37:"left",38:"up",39:"right",40:"down",45:"insert",46:"delete",48:"0",49:"1",50:"2",51:"3",52:"4",53:"5",54:"6",55:"7",56:"8",57:"9",59:"semicolon",61:"equals",65:"a",66:"b",67:"c",68:"d",69:"e",70:"f",71:"g",72:"h",73:"i",74:"j",75:"k",76:"l",77:"m",78:"n",
79:"o",80:"p",81:"q",82:"r",83:"s",84:"t",85:"u",86:"v",87:"w",88:"x",89:"y",90:"z",93:"context",96:"num-0",97:"num-1",98:"num-2",99:"num-3",100:"num-4",101:"num-5",102:"num-6",103:"num-7",104:"num-8",105:"num-9",106:"num-multiply",107:"num-plus",109:"num-minus",110:"num-period",111:"num-division",112:"f1",113:"f2",114:"f3",115:"f4",116:"f5",117:"f6",118:"f7",119:"f8",120:"f9",121:"f10",122:"f11",123:"f12",186:"semicolon",187:"equals",189:"dash",188:",",190:".",191:"/",192:"`",219:"open-square-bracket",
220:"\\",221:"close-square-bracket",222:"single-quote",224:"win"};_ds.f=bZ.prototype;_ds.f.keyCode=function(a){this.m=a;return this};_ds.f.key=function(a){this.o=a;return this};_ds.f.shiftKey=function(a){this.O=a;return this};_ds.f.altKey=function(a){this.h=a;return this};_ds.f.ctrlKey=function(a){this.j=a;return this};_ds.f.metaKey=function(a){this.C=a;return this};_ds.f.target=function(a){this.J=a;return this};_ds.xb(dZ,_ds.bh);
var rZ=[27,112,113,114,115,116,117,118,119,120,121,122,123,19],sZ="color date datetime datetime-local email month number password search tel text time url week".split(" "),gZ;_ds.xb(tZ,_ds.Tm);_ds.f=tZ.prototype;_ds.f.G_=function(){return this.G};_ds.f.yha=function(a){this.J=a};_ds.f.H_=function(){return this.J};_ds.f.vha=function(a){this.F=a};_ds.f.B_=function(){return this.F};_ds.f.Kla=function(a){this.L=a};_ds.f.b4=function(){return this.L};_ds.f.xha=function(a){this.N=a};_ds.f.Fk=_ds.la(1);
_ds.f.zpa=function(a){fZ(this.j,kZ(0,arguments))};_ds.f.Fda=function(a){return qZ(this,this.j,kZ(0,arguments))};_ds.f.cka=function(a){this.K=WY(a)};_ds.f.B2=function(){return _ds.Zb(this.K)};
_ds.f.ra=function(){tZ.Ba.ra.call(this);this.j={};_ds.zh(this.h,"keydown",this.wj,!1,this);_ds.zh(this.h,"synthetic-keydown",this.Bj,!1,this);_ds.qJ&&(_ds.zh(this.h,"keypress",this.Dj,!1,this),_ds.zh(this.h,"synthetic-keypress",this.Ej,!1,this));_ds.zh(this.h,"keyup",this.xj,!1,this);_ds.zh(this.h,"synthetic-keyup",this.Cj,!1,this);this.h=null};_ds.f.YJ=function(a){return"shortcut_"+a};_ds.f.xj=function(a){a=cZ(a);_ds.Ln&&lZ(this,a);_ds.qJ&&!this.o&&mZ(a)&&pZ(this,a,!0)};
_ds.f.Cj=function(a){a=a.j();_ds.Ln&&lZ(this,a);_ds.qJ&&!this.o&&mZ(a)&&pZ(this,a,!0)};_ds.f.Dj=function(a){a=cZ(a);32<a.h&&mZ(a)&&(this.o=!0)};_ds.f.Ej=function(a){a=a.j();32<a.h&&mZ(a)&&(this.o=!0)};_ds.f.wj=function(a){pZ(this,cZ(a))};_ds.f.Bj=function(a){pZ(this,a.j())};
_ds.LB.prototype.Pk=_ds.ma(2,function(a,b){for(var c=[],d=1;d<arguments.length;++d)c[d-1]=arguments[d];var e=this;this.j&&this.j.dispose();a&&(this.j=new tZ(document),this.j.G=!1,this.j.Fk.apply(this.j,[a].concat(_ds.ya(c))),this.eventHandler.listen(this.j,"shortcut",function(g){var h=document;h.activeElement&&h.activeElement===h.body&&(g.preventDefault(),_ds.KB(e,!0))}))});tZ.prototype.Fk=_ds.ma(1,function(a,b){nZ(this.j,kZ(1,arguments),a)});var IZ=/[ .()<>{}\[\]\/:,]+/,KZ=0;_ds.v(BZ,_ds.R);BZ.prototype.disconnectedCallback=function(){_ds.Fl(this.m);this.j&&(this.j.dispose(),this.j=null)};BZ.prototype.attributeChangedCallback=function(a,b,c){switch(a){case "project-scope":this.K=c||"";this.j&&(this.j.L=this.K);break;case "url-scoped":this.F=c;break;case "disabled":this.G&&(this.G.disabled=null!==c)}};
BZ.prototype.connectedCallback=function(){if(this.o=this.querySelector("form")){this.G=this.o.querySelector(".devsite-search-query");this.Y=this.o.querySelector(".devsite-search-button[search-open]");this.R=this.querySelector(".devsite-search-button[search-close]");var a="devsite-search-popout-container-id-"+ ++KZ;this.G.setAttribute("aria-controls",a);this.D=_ds.ol(vZ,{id:a});this.C=this.D.querySelector(".devsite-suggest-results-container");this.o.appendChild(this.D);this.hasAttribute("project-scope")&&
(this.K=this.getAttribute("project-scope"));this.hasAttribute("url-scoped")&&(this.F=this.getAttribute("url-scoped"));this.o&&this.G&&(this.j=new _ds.LB(this,this.o,this.G),this.j.o=!0,this.j.Ag=this.hasAttribute("enable-query-completion"),this.j.Ke=!0,this.j.zg=!0,this.j.Bg=!0,this.j.L=this.K,this.j.o=this.hasAttribute("enable-suggestions"));HZ(this)}};BZ.prototype.$=function(a){var b=this;_ds.gl(a.target,function(c){return c===b})||FZ(this,!1)};
_ds.n.Object.defineProperties(BZ,{observedAttributes:{configurable:!0,enumerable:!0,get:function(){return["project-scope","url-scoped","disabled"]}}});BZ.prototype.connectedCallback=BZ.prototype.connectedCallback;BZ.prototype.attributeChangedCallback=BZ.prototype.attributeChangedCallback;BZ.prototype.disconnectedCallback=BZ.prototype.disconnectedCallback;try{window.customElements.define("devsite-search",BZ)}catch(a){console.warn("devsite.app.customElement.DevsiteSearch",a)};})(_ds_www);
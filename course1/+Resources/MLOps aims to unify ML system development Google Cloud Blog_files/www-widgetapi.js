(function(){/*

 Copyright The Closure Library Authors.
 SPDX-License-Identifier: Apache-2.0
*/
'use strict';var n;function aa(a){var b=0;return function(){return b<a.length?{done:!1,value:a[b++]}:{done:!0}}}
var ba="function"==typeof Object.defineProperties?Object.defineProperty:function(a,b,c){if(a==Array.prototype||a==Object.prototype)return a;a[b]=c.value;return a};
function ca(a){a=["object"==typeof globalThis&&globalThis,a,"object"==typeof window&&window,"object"==typeof self&&self,"object"==typeof global&&global];for(var b=0;b<a.length;++b){var c=a[b];if(c&&c.Math==Math)return c}throw Error("Cannot find global object");}
var da=ca(this);function t(a,b){if(b)a:{var c=da;a=a.split(".");for(var d=0;d<a.length-1;d++){var e=a[d];if(!(e in c))break a;c=c[e]}a=a[a.length-1];d=c[a];b=b(d);b!=d&&null!=b&&ba(c,a,{configurable:!0,writable:!0,value:b})}}
t("Symbol",function(a){function b(f){if(this instanceof b)throw new TypeError("Symbol is not a constructor");return new c(d+(f||"")+"_"+e++,f)}
function c(f,g){this.h=f;ba(this,"description",{configurable:!0,writable:!0,value:g})}
if(a)return a;c.prototype.toString=function(){return this.h};
var d="jscomp_symbol_"+(1E9*Math.random()>>>0)+"_",e=0;return b});
t("Symbol.iterator",function(a){if(a)return a;a=Symbol("Symbol.iterator");for(var b="Array Int8Array Uint8Array Uint8ClampedArray Int16Array Uint16Array Int32Array Uint32Array Float32Array Float64Array".split(" "),c=0;c<b.length;c++){var d=da[b[c]];"function"===typeof d&&"function"!=typeof d.prototype[a]&&ba(d.prototype,a,{configurable:!0,writable:!0,value:function(){return ea(aa(this))}})}return a});
function ea(a){a={next:a};a[Symbol.iterator]=function(){return this};
return a}
function u(a){var b="undefined"!=typeof Symbol&&Symbol.iterator&&a[Symbol.iterator];return b?b.call(a):{next:aa(a)}}
var fa="function"==typeof Object.create?Object.create:function(a){function b(){}
b.prototype=a;return new b},ha;
if("function"==typeof Object.setPrototypeOf)ha=Object.setPrototypeOf;else{var ia;a:{var ja={a:!0},ka={};try{ka.__proto__=ja;ia=ka.a;break a}catch(a){}ia=!1}ha=ia?function(a,b){a.__proto__=b;if(a.__proto__!==b)throw new TypeError(a+" is not extensible");return a}:null}var la=ha;
function v(a,b){a.prototype=fa(b.prototype);a.prototype.constructor=a;if(la)la(a,b);else for(var c in b)if("prototype"!=c)if(Object.defineProperties){var d=Object.getOwnPropertyDescriptor(b,c);d&&Object.defineProperty(a,c,d)}else a[c]=b[c];a.G=b.prototype}
function ma(){this.o=!1;this.l=null;this.i=void 0;this.h=1;this.m=this.s=0;this.B=this.j=null}
function na(a){if(a.o)throw new TypeError("Generator is already running");a.o=!0}
ma.prototype.v=function(a){this.i=a};
function oa(a,b){a.j={ja:b,ka:!0};a.h=a.s||a.m}
ma.prototype.return=function(a){this.j={return:a};this.h=this.m};
function w(a,b,c){a.h=c;return{value:b}}
ma.prototype.u=function(a){this.h=a};
function pa(a,b,c){a.s=b;void 0!=c&&(a.m=c)}
function qa(a){a.s=0;var b=a.j.ja;a.j=null;return b}
function ra(a){a.B=[a.j];a.s=0;a.m=0}
function sa(a){var b=a.B.splice(0)[0];(b=a.j=a.j||b)?b.ka?a.h=a.s||a.m:void 0!=b.u&&a.m<b.u?(a.h=b.u,a.j=null):a.h=a.m:a.h=0}
function ta(a){this.h=new ma;this.i=a}
function ua(a,b){na(a.h);var c=a.h.l;if(c)return va(a,"return"in c?c["return"]:function(d){return{value:d,done:!0}},b,a.h.return);
a.h.return(b);return wa(a)}
function va(a,b,c,d){try{var e=b.call(a.h.l,c);if(!(e instanceof Object))throw new TypeError("Iterator result "+e+" is not an object");if(!e.done)return a.h.o=!1,e;var f=e.value}catch(g){return a.h.l=null,oa(a.h,g),wa(a)}a.h.l=null;d.call(a.h,f);return wa(a)}
function wa(a){for(;a.h.h;)try{var b=a.i(a.h);if(b)return a.h.o=!1,{value:b.value,done:!1}}catch(c){a.h.i=void 0,oa(a.h,c)}a.h.o=!1;if(a.h.j){b=a.h.j;a.h.j=null;if(b.ka)throw b.ja;return{value:b.return,done:!0}}return{value:void 0,done:!0}}
function ya(a){this.next=function(b){na(a.h);a.h.l?b=va(a,a.h.l.next,b,a.h.v):(a.h.v(b),b=wa(a));return b};
this.throw=function(b){na(a.h);a.h.l?b=va(a,a.h.l["throw"],b,a.h.v):(oa(a.h,b),b=wa(a));return b};
this.return=function(b){return ua(a,b)};
this[Symbol.iterator]=function(){return this}}
function x(a,b){b=new ya(new ta(b));la&&a.prototype&&la(b,a.prototype);return b}
t("Reflect.setPrototypeOf",function(a){return a?a:la?function(b,c){try{return la(b,c),!0}catch(d){return!1}}:null});
function za(a,b){return Object.prototype.hasOwnProperty.call(a,b)}
t("WeakMap",function(a){function b(k){this.h=(h+=Math.random()+1).toString();if(k){k=u(k);for(var l;!(l=k.next()).done;)l=l.value,this.set(l[0],l[1])}}
function c(){}
function d(k){var l=typeof k;return"object"===l&&null!==k||"function"===l}
function e(k){if(!za(k,g)){var l=new c;ba(k,g,{value:l})}}
function f(k){var l=Object[k];l&&(Object[k]=function(m){if(m instanceof c)return m;Object.isExtensible(m)&&e(m);return l(m)})}
if(function(){if(!a||!Object.seal)return!1;try{var k=Object.seal({}),l=Object.seal({}),m=new a([[k,2],[l,3]]);if(2!=m.get(k)||3!=m.get(l))return!1;m.delete(k);m.set(l,4);return!m.has(k)&&4==m.get(l)}catch(p){return!1}}())return a;
var g="$jscomp_hidden_"+Math.random();f("freeze");f("preventExtensions");f("seal");var h=0;b.prototype.set=function(k,l){if(!d(k))throw Error("Invalid WeakMap key");e(k);if(!za(k,g))throw Error("WeakMap key fail: "+k);k[g][this.h]=l;return this};
b.prototype.get=function(k){return d(k)&&za(k,g)?k[g][this.h]:void 0};
b.prototype.has=function(k){return d(k)&&za(k,g)&&za(k[g],this.h)};
b.prototype.delete=function(k){return d(k)&&za(k,g)&&za(k[g],this.h)?delete k[g][this.h]:!1};
return b});
t("Map",function(a){function b(){var h={};return h.previous=h.next=h.head=h}
function c(h,k){var l=h.h;return ea(function(){if(l){for(;l.head!=h.h;)l=l.previous;for(;l.next!=l.head;)return l=l.next,{done:!1,value:k(l)};l=null}return{done:!0,value:void 0}})}
function d(h,k){var l=k&&typeof k;"object"==l||"function"==l?f.has(k)?l=f.get(k):(l=""+ ++g,f.set(k,l)):l="p_"+k;var m=h.data_[l];if(m&&za(h.data_,l))for(h=0;h<m.length;h++){var p=m[h];if(k!==k&&p.key!==p.key||k===p.key)return{id:l,list:m,index:h,A:p}}return{id:l,list:m,index:-1,A:void 0}}
function e(h){this.data_={};this.h=b();this.size=0;if(h){h=u(h);for(var k;!(k=h.next()).done;)k=k.value,this.set(k[0],k[1])}}
if(function(){if(!a||"function"!=typeof a||!a.prototype.entries||"function"!=typeof Object.seal)return!1;try{var h=Object.seal({x:4}),k=new a(u([[h,"s"]]));if("s"!=k.get(h)||1!=k.size||k.get({x:4})||k.set({x:4},"t")!=k||2!=k.size)return!1;var l=k.entries(),m=l.next();if(m.done||m.value[0]!=h||"s"!=m.value[1])return!1;m=l.next();return m.done||4!=m.value[0].x||"t"!=m.value[1]||!l.next().done?!1:!0}catch(p){return!1}}())return a;
var f=new WeakMap;e.prototype.set=function(h,k){h=0===h?0:h;var l=d(this,h);l.list||(l.list=this.data_[l.id]=[]);l.A?l.A.value=k:(l.A={next:this.h,previous:this.h.previous,head:this.h,key:h,value:k},l.list.push(l.A),this.h.previous.next=l.A,this.h.previous=l.A,this.size++);return this};
e.prototype.delete=function(h){h=d(this,h);return h.A&&h.list?(h.list.splice(h.index,1),h.list.length||delete this.data_[h.id],h.A.previous.next=h.A.next,h.A.next.previous=h.A.previous,h.A.head=null,this.size--,!0):!1};
e.prototype.clear=function(){this.data_={};this.h=this.h.previous=b();this.size=0};
e.prototype.has=function(h){return!!d(this,h).A};
e.prototype.get=function(h){return(h=d(this,h).A)&&h.value};
e.prototype.entries=function(){return c(this,function(h){return[h.key,h.value]})};
e.prototype.keys=function(){return c(this,function(h){return h.key})};
e.prototype.values=function(){return c(this,function(h){return h.value})};
e.prototype.forEach=function(h,k){for(var l=this.entries(),m;!(m=l.next()).done;)m=m.value,h.call(k,m[1],m[0],this)};
e.prototype[Symbol.iterator]=e.prototype.entries;var g=0;return e});
function Aa(a,b,c){if(null==a)throw new TypeError("The 'this' value for String.prototype."+c+" must not be null or undefined");if(b instanceof RegExp)throw new TypeError("First argument to String.prototype."+c+" must not be a regular expression");return a+""}
t("String.prototype.endsWith",function(a){return a?a:function(b,c){var d=Aa(this,b,"endsWith");b+="";void 0===c&&(c=d.length);c=Math.max(0,Math.min(c|0,d.length));for(var e=b.length;0<e&&0<c;)if(d[--c]!=b[--e])return!1;return 0>=e}});
t("String.prototype.startsWith",function(a){return a?a:function(b,c){var d=Aa(this,b,"startsWith");b+="";var e=d.length,f=b.length;c=Math.max(0,Math.min(c|0,d.length));for(var g=0;g<f&&c<e;)if(d[c++]!=b[g++])return!1;return g>=f}});
t("Object.setPrototypeOf",function(a){return a||la});
var Ba="function"==typeof Object.assign?Object.assign:function(a,b){for(var c=1;c<arguments.length;c++){var d=arguments[c];if(d)for(var e in d)za(d,e)&&(a[e]=d[e])}return a};
t("Object.assign",function(a){return a||Ba});
t("Promise",function(a){function b(g){this.h=0;this.j=void 0;this.i=[];this.o=!1;var h=this.l();try{g(h.resolve,h.reject)}catch(k){h.reject(k)}}
function c(){this.h=null}
function d(g){return g instanceof b?g:new b(function(h){h(g)})}
if(a)return a;c.prototype.i=function(g){if(null==this.h){this.h=[];var h=this;this.j(function(){h.m()})}this.h.push(g)};
var e=da.setTimeout;c.prototype.j=function(g){e(g,0)};
c.prototype.m=function(){for(;this.h&&this.h.length;){var g=this.h;this.h=[];for(var h=0;h<g.length;++h){var k=g[h];g[h]=null;try{k()}catch(l){this.l(l)}}}this.h=null};
c.prototype.l=function(g){this.j(function(){throw g;})};
b.prototype.l=function(){function g(l){return function(m){k||(k=!0,l.call(h,m))}}
var h=this,k=!1;return{resolve:g(this.U),reject:g(this.m)}};
b.prototype.U=function(g){if(g===this)this.m(new TypeError("A Promise cannot resolve to itself"));else if(g instanceof b)this.da(g);else{a:switch(typeof g){case "object":var h=null!=g;break a;case "function":h=!0;break a;default:h=!1}h?this.P(g):this.s(g)}};
b.prototype.P=function(g){var h=void 0;try{h=g.then}catch(k){this.m(k);return}"function"==typeof h?this.sa(h,g):this.s(g)};
b.prototype.m=function(g){this.v(2,g)};
b.prototype.s=function(g){this.v(1,g)};
b.prototype.v=function(g,h){if(0!=this.h)throw Error("Cannot settle("+g+", "+h+"): Promise already settled in state"+this.h);this.h=g;this.j=h;2===this.h&&this.V();this.B()};
b.prototype.V=function(){var g=this;e(function(){if(g.M()){var h=da.console;"undefined"!==typeof h&&h.error(g.j)}},1)};
b.prototype.M=function(){if(this.o)return!1;var g=da.CustomEvent,h=da.Event,k=da.dispatchEvent;if("undefined"===typeof k)return!0;"function"===typeof g?g=new g("unhandledrejection",{cancelable:!0}):"function"===typeof h?g=new h("unhandledrejection",{cancelable:!0}):(g=da.document.createEvent("CustomEvent"),g.initCustomEvent("unhandledrejection",!1,!0,g));g.promise=this;g.reason=this.j;return k(g)};
b.prototype.B=function(){if(null!=this.i){for(var g=0;g<this.i.length;++g)f.i(this.i[g]);this.i=null}};
var f=new c;b.prototype.da=function(g){var h=this.l();g.X(h.resolve,h.reject)};
b.prototype.sa=function(g,h){var k=this.l();try{g.call(h,k.resolve,k.reject)}catch(l){k.reject(l)}};
b.prototype.then=function(g,h){function k(r,q){return"function"==typeof r?function(y){try{l(r(y))}catch(C){m(C)}}:q}
var l,m,p=new b(function(r,q){l=r;m=q});
this.X(k(g,l),k(h,m));return p};
b.prototype.catch=function(g){return this.then(void 0,g)};
b.prototype.X=function(g,h){function k(){switch(l.h){case 1:g(l.j);break;case 2:h(l.j);break;default:throw Error("Unexpected state: "+l.h);}}
var l=this;null==this.i?f.i(k):this.i.push(k);this.o=!0};
b.resolve=d;b.reject=function(g){return new b(function(h,k){k(g)})};
b.race=function(g){return new b(function(h,k){for(var l=u(g),m=l.next();!m.done;m=l.next())d(m.value).X(h,k)})};
b.all=function(g){var h=u(g),k=h.next();return k.done?d([]):new b(function(l,m){function p(y){return function(C){r[y]=C;q--;0==q&&l(r)}}
var r=[],q=0;do r.push(void 0),q++,d(k.value).X(p(r.length-1),m),k=h.next();while(!k.done)})};
return b});
function Ca(a,b){a instanceof String&&(a+="");var c=0,d=!1,e={next:function(){if(!d&&c<a.length){var f=c++;return{value:b(f,a[f]),done:!1}}d=!0;return{done:!0,value:void 0}}};
e[Symbol.iterator]=function(){return e};
return e}
t("Array.prototype.entries",function(a){return a?a:function(){return Ca(this,function(b,c){return[b,c]})}});
t("Object.entries",function(a){return a?a:function(b){var c=[],d;for(d in b)za(b,d)&&c.push([d,b[d]]);return c}});
t("Array.prototype.keys",function(a){return a?a:function(){return Ca(this,function(b){return b})}});
t("Array.prototype.values",function(a){return a?a:function(){return Ca(this,function(b,c){return c})}});
t("Array.from",function(a){return a?a:function(b,c,d){c=null!=c?c:function(h){return h};
var e=[],f="undefined"!=typeof Symbol&&Symbol.iterator&&b[Symbol.iterator];if("function"==typeof f){b=f.call(b);for(var g=0;!(f=b.next()).done;)e.push(c.call(d,f.value,g++))}else for(f=b.length,g=0;g<f;g++)e.push(c.call(d,b[g],g));return e}});
t("Number.isNaN",function(a){return a?a:function(b){return"number"===typeof b&&isNaN(b)}});
t("Number.MAX_SAFE_INTEGER",function(){return 9007199254740991});
t("Object.is",function(a){return a?a:function(b,c){return b===c?0!==b||1/b===1/c:b!==b&&c!==c}});
t("Array.prototype.includes",function(a){return a?a:function(b,c){var d=this;d instanceof String&&(d=String(d));var e=d.length;c=c||0;for(0>c&&(c=Math.max(c+e,0));c<e;c++){var f=d[c];if(f===b||Object.is(f,b))return!0}return!1}});
t("String.prototype.includes",function(a){return a?a:function(b,c){return-1!==Aa(this,b,"includes").indexOf(b,c||0)}});
t("Set",function(a){function b(c){this.h=new Map;if(c){c=u(c);for(var d;!(d=c.next()).done;)this.add(d.value)}this.size=this.h.size}
if(function(){if(!a||"function"!=typeof a||!a.prototype.entries||"function"!=typeof Object.seal)return!1;try{var c=Object.seal({x:4}),d=new a(u([c]));if(!d.has(c)||1!=d.size||d.add(c)!=d||1!=d.size||d.add({x:4})!=d||2!=d.size)return!1;var e=d.entries(),f=e.next();if(f.done||f.value[0]!=c||f.value[1]!=c)return!1;f=e.next();return f.done||f.value[0]==c||4!=f.value[0].x||f.value[1]!=f.value[0]?!1:e.next().done}catch(g){return!1}}())return a;
b.prototype.add=function(c){c=0===c?0:c;this.h.set(c,c);this.size=this.h.size;return this};
b.prototype.delete=function(c){c=this.h.delete(c);this.size=this.h.size;return c};
b.prototype.clear=function(){this.h.clear();this.size=0};
b.prototype.has=function(c){return this.h.has(c)};
b.prototype.entries=function(){return this.h.entries()};
b.prototype.values=function(){return this.h.values()};
b.prototype.keys=b.prototype.values;b.prototype[Symbol.iterator]=b.prototype.values;b.prototype.forEach=function(c,d){var e=this;this.h.forEach(function(f){return c.call(d,f,f,e)})};
return b});
var z=this||self;function B(a,b){a=a.split(".");b=b||z;for(var c=0;c<a.length;c++)if(b=b[a[c]],null==b)return null;return b}
function Da(){}
function Ea(a){var b=typeof a;b="object"!=b?b:a?Array.isArray(a)?"array":b:"null";return"array"==b||"object"==b&&"number"==typeof a.length}
function D(a){var b=typeof a;return"object"==b&&null!=a||"function"==b}
function Fa(a){return Object.prototype.hasOwnProperty.call(a,Ga)&&a[Ga]||(a[Ga]=++Ha)}
var Ga="closure_uid_"+(1E9*Math.random()>>>0),Ha=0;function Ia(a,b,c){return a.call.apply(a.bind,arguments)}
function Ja(a,b,c){if(!a)throw Error();if(2<arguments.length){var d=Array.prototype.slice.call(arguments,2);return function(){var e=Array.prototype.slice.call(arguments);Array.prototype.unshift.apply(e,d);return a.apply(b,e)}}return function(){return a.apply(b,arguments)}}
function Ka(a,b,c){Function.prototype.bind&&-1!=Function.prototype.bind.toString().indexOf("native code")?Ka=Ia:Ka=Ja;return Ka.apply(null,arguments)}
function F(a,b){a=a.split(".");var c=z;a[0]in c||"undefined"==typeof c.execScript||c.execScript("var "+a[0]);for(var d;a.length&&(d=a.shift());)a.length||void 0===b?c[d]&&c[d]!==Object.prototype[d]?c=c[d]:c=c[d]={}:c[d]=b}
function G(a,b){function c(){}
c.prototype=b.prototype;a.G=b.prototype;a.prototype=new c;a.prototype.constructor=a;a.Ya=function(d,e,f){for(var g=Array(arguments.length-2),h=2;h<arguments.length;h++)g[h-2]=arguments[h];return b.prototype[e].apply(d,g)}}
function La(a){return a}
;function Ma(a,b){if(Error.captureStackTrace)Error.captureStackTrace(this,Ma);else{var c=Error().stack;c&&(this.stack=c)}a&&(this.message=String(a));b&&(this.wa=b)}
G(Ma,Error);Ma.prototype.name="CustomError";function Na(a){a=a.url;var b=/[?&]dsh=1(&|$)/.test(a);this.j=!b&&/[?&]ae=1(&|$)/.test(a);this.l=!b&&/[?&]ae=2(&|$)/.test(a);if((this.h=/[?&]adurl=([^&]*)/.exec(a))&&this.h[1]){try{var c=decodeURIComponent(this.h[1])}catch(d){c=null}this.i=c}}
;function Oa(a){var b=!1,c;return function(){b||(c=a(),b=!0);return c}}
;var Pa=Array.prototype.indexOf?function(a,b){return Array.prototype.indexOf.call(a,b,void 0)}:function(a,b){if("string"===typeof a)return"string"!==typeof b||1!=b.length?-1:a.indexOf(b,0);
for(var c=0;c<a.length;c++)if(c in a&&a[c]===b)return c;return-1},H=Array.prototype.forEach?function(a,b,c){Array.prototype.forEach.call(a,b,c)}:function(a,b,c){for(var d=a.length,e="string"===typeof a?a.split(""):a,f=0;f<d;f++)f in e&&b.call(c,e[f],f,a)},Qa=Array.prototype.reduce?function(a,b,c){return Array.prototype.reduce.call(a,b,c)}:function(a,b,c){var d=c;
H(a,function(e,f){d=b.call(void 0,d,e,f,a)});
return d};
function Ra(a,b){a:{for(var c=a.length,d="string"===typeof a?a.split(""):a,e=0;e<c;e++)if(e in d&&b.call(void 0,d[e],e,a)){b=e;break a}b=-1}return 0>b?null:"string"===typeof a?a.charAt(b):a[b]}
function Sa(a,b){b=Pa(a,b);var c;(c=0<=b)&&Array.prototype.splice.call(a,b,1);return c}
function Ta(a){return Array.prototype.concat.apply([],arguments)}
function Ua(a){var b=a.length;if(0<b){for(var c=Array(b),d=0;d<b;d++)c[d]=a[d];return c}return[]}
function Va(a,b){for(var c=1;c<arguments.length;c++){var d=arguments[c];if(Ea(d)){var e=a.length||0,f=d.length||0;a.length=e+f;for(var g=0;g<f;g++)a[e+g]=d[g]}else a.push(d)}}
;function Wa(a,b){for(var c in a)b.call(void 0,a[c],c,a)}
function Xa(a){var b=Ya,c;for(c in b)if(a.call(void 0,b[c],c,b))return c}
function Za(a,b){for(var c in a)if(!(c in b)||a[c]!==b[c])return!1;for(var d in b)if(!(d in a))return!1;return!0}
function $a(a){if(!a||"object"!==typeof a)return a;if("function"===typeof a.clone)return a.clone();if("undefined"!==typeof Map&&a instanceof Map)return new Map(a);if("undefined"!==typeof Set&&a instanceof Set)return new Set(a);var b=Array.isArray(a)?[]:"function"!==typeof ArrayBuffer||"function"!==typeof ArrayBuffer.isView||!ArrayBuffer.isView(a)||a instanceof DataView?{}:new a.constructor(a.length),c;for(c in a)b[c]=$a(a[c]);return b}
var ab="constructor hasOwnProperty isPrototypeOf propertyIsEnumerable toLocaleString toString valueOf".split(" ");function bb(a,b){for(var c,d,e=1;e<arguments.length;e++){d=arguments[e];for(c in d)a[c]=d[c];for(var f=0;f<ab.length;f++)c=ab[f],Object.prototype.hasOwnProperty.call(d,c)&&(a[c]=d[c])}}
;var cb;var db=String.prototype.trim?function(a){return a.trim()}:function(a){return/^[\s\xa0]*([\s\S]*?)[\s\xa0]*$/.exec(a)[1]},eb=/&/g,fb=/</g,gb=/>/g,hb=/"/g,ib=/'/g,jb=/\x00/g,kb=/[\x00&<>"']/;var lb;a:{var mb=z.navigator;if(mb){var nb=mb.userAgent;if(nb){lb=nb;break a}}lb=""}function I(a){return-1!=lb.indexOf(a)}
;function ob(a){this.h=pb===pb?a:""}
ob.prototype.toString=function(){return this.h.toString()};
var pb={};var qb=/^(?:([^:/?#.]+):)?(?:\/\/(?:([^\\/?#]*)@)?([^\\/?#]*?)(?::([0-9]+))?(?=[\\/?#]|$))?([^?#]+)?(?:\?([^#]*))?(?:#([\s\S]*))?$/;function rb(a){return a?decodeURI(a):a}
function sb(a){return rb(a.match(qb)[3]||null)}
function tb(a){var b=a.match(qb);a=b[1];var c=b[2],d=b[3];b=b[4];var e="";a&&(e+=a+":");d&&(e+="//",c&&(e+=c+"@"),e+=d,b&&(e+=":"+b));return e}
function ub(a,b,c){if(Array.isArray(b))for(var d=0;d<b.length;d++)ub(a,String(b[d]),c);else null!=b&&c.push(a+(""===b?"":"="+encodeURIComponent(String(b))))}
function vb(a){var b=[],c;for(c in a)ub(c,a[c],b);return b.join("&")}
var wb=/#|$/;function xb(a,b){var c=a.search(wb);a:{var d=0;for(var e=b.length;0<=(d=a.indexOf(b,d))&&d<c;){var f=a.charCodeAt(d-1);if(38==f||63==f)if(f=a.charCodeAt(d+e),!f||61==f||38==f||35==f)break a;d+=e+1}d=-1}if(0>d)return null;e=a.indexOf("&",d);if(0>e||e>c)e=c;d+=b.length+1;return decodeURIComponent(a.substr(d,e-d).replace(/\+/g," "))}
;function J(a,b){var c=void 0;return new (c||(c=Promise))(function(d,e){function f(k){try{h(b.next(k))}catch(l){e(l)}}
function g(k){try{h(b["throw"](k))}catch(l){e(l)}}
function h(k){k.done?d(k.value):(new c(function(l){l(k.value)})).then(f,g)}
h((b=b.apply(a,void 0)).next())})}
;function yb(){return I("iPhone")&&!I("iPod")&&!I("iPad")}
;function zb(a){zb[" "](a);return a}
zb[" "]=Da;var Ab=I("Opera"),Bb=I("Trident")||I("MSIE"),Cb=I("Edge"),Db=I("Gecko")&&!(-1!=lb.toLowerCase().indexOf("webkit")&&!I("Edge"))&&!(I("Trident")||I("MSIE"))&&!I("Edge"),Eb=-1!=lb.toLowerCase().indexOf("webkit")&&!I("Edge");function Fb(){var a=z.document;return a?a.documentMode:void 0}
var Gb;a:{var Hb="",Ib=function(){var a=lb;if(Db)return/rv:([^\);]+)(\)|;)/.exec(a);if(Cb)return/Edge\/([\d\.]+)/.exec(a);if(Bb)return/\b(?:MSIE|rv)[: ]([^\);]+)(\)|;)/.exec(a);if(Eb)return/WebKit\/(\S+)/.exec(a);if(Ab)return/(?:Version)[ \/]?(\S+)/.exec(a)}();
Ib&&(Hb=Ib?Ib[1]:"");if(Bb){var Jb=Fb();if(null!=Jb&&Jb>parseFloat(Hb)){Gb=String(Jb);break a}}Gb=Hb}var Kb=Gb,Lb;if(z.document&&Bb){var Mb=Fb();Lb=Mb?Mb:parseInt(Kb,10)||void 0}else Lb=void 0;var Nb=Lb;var Ob=yb()||I("iPod"),Pb=I("iPad"),Qb=I("Safari")&&!((I("Chrome")||I("CriOS"))&&!I("Edge")||I("Coast")||I("Opera")||I("Edge")||I("Edg/")||I("OPR")||I("Firefox")||I("FxiOS")||I("Silk")||I("Android"))&&!(yb()||I("iPad")||I("iPod"));var Rb={},Sb=null;Object.freeze([]);var K=window;function Tb(a,b){this.width=a;this.height=b}
n=Tb.prototype;n.clone=function(){return new Tb(this.width,this.height)};
n.aspectRatio=function(){return this.width/this.height};
n.isEmpty=function(){return!(this.width*this.height)};
n.ceil=function(){this.width=Math.ceil(this.width);this.height=Math.ceil(this.height);return this};
n.floor=function(){this.width=Math.floor(this.width);this.height=Math.floor(this.height);return this};
n.round=function(){this.width=Math.round(this.width);this.height=Math.round(this.height);return this};function Ub(){var a=document;var b="IFRAME";"application/xhtml+xml"===a.contentType&&(b=b.toLowerCase());return a.createElement(b)}
function Vb(a,b){for(var c=0;a;){if(b(a))return a;a=a.parentNode;c++}return null}
;function Wb(a){var b=Xb;if(b)for(var c in b)Object.prototype.hasOwnProperty.call(b,c)&&a.call(void 0,b[c],c,b)}
function Yb(){var a=[];Wb(function(b){a.push(b)});
return a}
var Xb={La:"allow-forms",Ma:"allow-modals",Na:"allow-orientation-lock",Oa:"allow-pointer-lock",Pa:"allow-popups",Qa:"allow-popups-to-escape-sandbox",Ra:"allow-presentation",Sa:"allow-same-origin",Ta:"allow-scripts",Ua:"allow-top-navigation",Va:"allow-top-navigation-by-user-activation"},Zb=Oa(function(){return Yb()});
function $b(){var a=Ub(),b={};H(Zb(),function(c){a.sandbox&&a.sandbox.supports&&a.sandbox.supports(c)&&(b[c]=!0)});
return b}
;var ac=(new Date).getTime();function bc(a){if(!a)return"";a=a.split("#")[0].split("?")[0];a=a.toLowerCase();0==a.indexOf("//")&&(a=window.location.protocol+a);/^[\w\-]*:\/\//.test(a)||(a=window.location.href);var b=a.substring(a.indexOf("://")+3),c=b.indexOf("/");-1!=c&&(b=b.substring(0,c));c=a.substring(0,a.indexOf("://"));if(!c)throw Error("URI is missing protocol: "+a);if("http"!==c&&"https"!==c&&"chrome-extension"!==c&&"moz-extension"!==c&&"file"!==c&&"android-app"!==c&&"chrome-search"!==c&&"chrome-untrusted"!==c&&"chrome"!==
c&&"app"!==c&&"devtools"!==c)throw Error("Invalid URI scheme in origin: "+c);a="";var d=b.indexOf(":");if(-1!=d){var e=b.substring(d+1);b=b.substring(0,d);if("http"===c&&"80"!==e||"https"===c&&"443"!==e)a=":"+e}return c+"://"+b+a}
;function cc(){function a(){e[0]=1732584193;e[1]=4023233417;e[2]=2562383102;e[3]=271733878;e[4]=3285377520;m=l=0}
function b(p){for(var r=g,q=0;64>q;q+=4)r[q/4]=p[q]<<24|p[q+1]<<16|p[q+2]<<8|p[q+3];for(q=16;80>q;q++)p=r[q-3]^r[q-8]^r[q-14]^r[q-16],r[q]=(p<<1|p>>>31)&4294967295;p=e[0];var y=e[1],C=e[2],A=e[3],Q=e[4];for(q=0;80>q;q++){if(40>q)if(20>q){var R=A^y&(C^A);var E=1518500249}else R=y^C^A,E=1859775393;else 60>q?(R=y&C|A&(y|C),E=2400959708):(R=y^C^A,E=3395469782);R=((p<<5|p>>>27)&4294967295)+R+Q+E+r[q]&4294967295;Q=A;A=C;C=(y<<30|y>>>2)&4294967295;y=p;p=R}e[0]=e[0]+p&4294967295;e[1]=e[1]+y&4294967295;e[2]=
e[2]+C&4294967295;e[3]=e[3]+A&4294967295;e[4]=e[4]+Q&4294967295}
function c(p,r){if("string"===typeof p){p=unescape(encodeURIComponent(p));for(var q=[],y=0,C=p.length;y<C;++y)q.push(p.charCodeAt(y));p=q}r||(r=p.length);q=0;if(0==l)for(;q+64<r;)b(p.slice(q,q+64)),q+=64,m+=64;for(;q<r;)if(f[l++]=p[q++],m++,64==l)for(l=0,b(f);q+64<r;)b(p.slice(q,q+64)),q+=64,m+=64}
function d(){var p=[],r=8*m;56>l?c(h,56-l):c(h,64-(l-56));for(var q=63;56<=q;q--)f[q]=r&255,r>>>=8;b(f);for(q=r=0;5>q;q++)for(var y=24;0<=y;y-=8)p[r++]=e[q]>>y&255;return p}
for(var e=[],f=[],g=[],h=[128],k=1;64>k;++k)h[k]=0;var l,m;a();return{reset:a,update:c,digest:d,xa:function(){for(var p=d(),r="",q=0;q<p.length;q++)r+="0123456789ABCDEF".charAt(Math.floor(p[q]/16))+"0123456789ABCDEF".charAt(p[q]%16);return r}}}
;function dc(a,b,c){var d=String(z.location.href);return d&&a&&b?[b,ec(bc(d),a,c||null)].join(" "):null}
function ec(a,b,c){var d=[],e=[];if(1==(Array.isArray(c)?2:1))return e=[b,a],H(d,function(h){e.push(h)}),fc(e.join(" "));
var f=[],g=[];H(c,function(h){g.push(h.key);f.push(h.value)});
c=Math.floor((new Date).getTime()/1E3);e=0==f.length?[c,b,a]:[f.join(":"),c,b,a];H(d,function(h){e.push(h)});
a=fc(e.join(" "));a=[c,a];0==g.length||a.push(g.join(""));return a.join("_")}
function fc(a){var b=cc();b.update(a);return b.xa().toLowerCase()}
;var gc={};function hc(a){this.h=a||{cookie:""}}
n=hc.prototype;n.isEnabled=function(){if(!z.navigator.cookieEnabled)return!1;if(!this.isEmpty())return!0;this.set("TESTCOOKIESENABLED","1",{ea:60});if("1"!==this.get("TESTCOOKIESENABLED"))return!1;this.remove("TESTCOOKIESENABLED");return!0};
n.set=function(a,b,c){var d=!1;if("object"===typeof c){var e=c.eb;d=c.secure||!1;var f=c.domain||void 0;var g=c.path||void 0;var h=c.ea}if(/[;=\s]/.test(a))throw Error('Invalid cookie name "'+a+'"');if(/[;\r\n]/.test(b))throw Error('Invalid cookie value "'+b+'"');void 0===h&&(h=-1);this.h.cookie=a+"="+b+(f?";domain="+f:"")+(g?";path="+g:"")+(0>h?"":0==h?";expires="+(new Date(1970,1,1)).toUTCString():";expires="+(new Date(Date.now()+1E3*h)).toUTCString())+(d?";secure":"")+(null!=e?";samesite="+e:"")};
n.get=function(a,b){for(var c=a+"=",d=(this.h.cookie||"").split(";"),e=0,f;e<d.length;e++){f=db(d[e]);if(0==f.lastIndexOf(c,0))return f.substr(c.length);if(f==a)return""}return b};
n.remove=function(a,b,c){var d=void 0!==this.get(a);this.set(a,"",{ea:0,path:b,domain:c});return d};
n.isEmpty=function(){return!this.h.cookie};
n.clear=function(){for(var a=(this.h.cookie||"").split(";"),b=[],c=[],d,e,f=0;f<a.length;f++)e=db(a[f]),d=e.indexOf("="),-1==d?(b.push(""),c.push(e)):(b.push(e.substring(0,d)),c.push(e.substring(d+1)));for(a=b.length-1;0<=a;a--)this.remove(b[a])};
var ic=new hc("undefined"==typeof document?null:document);function jc(a){return!!gc.FPA_SAMESITE_PHASE2_MOD||!(void 0===a||!a)}
function kc(a,b,c,d){(a=z[a])||(a=(new hc(document)).get(b));return a?dc(a,c,d):null}
function lc(a){var b=void 0===b?!1:b;var c=bc(String(z.location.href)),d=[];var e=b;e=void 0===e?!1:e;var f=z.__SAPISID||z.__APISID||z.__3PSAPISID||z.__OVERRIDE_SID;jc(e)&&(f=f||z.__1PSAPISID);if(f)e=!0;else{var g=new hc(document);f=g.get("SAPISID")||g.get("APISID")||g.get("__Secure-3PAPISID")||g.get("SID");jc(e)&&(f=f||g.get("__Secure-1PAPISID"));e=!!f}e&&(e=(c=0==c.indexOf("https:")||0==c.indexOf("chrome-extension:")||0==c.indexOf("moz-extension:"))?z.__SAPISID:z.__APISID,e||(e=new hc(document),
e=e.get(c?"SAPISID":"APISID")||e.get("__Secure-3PAPISID")),(e=e?dc(e,c?"SAPISIDHASH":"APISIDHASH",a):null)&&d.push(e),c&&jc(b)&&((b=kc("__1PSAPISID","__Secure-1PAPISID","SAPISID1PHASH",a))&&d.push(b),(a=kc("__3PSAPISID","__Secure-3PAPISID","SAPISID3PHASH",a))&&d.push(a)));return 0==d.length?null:d.join(" ")}
;function mc(){this.data_=[];this.h=-1}
mc.prototype.set=function(a,b){b=void 0===b?!0:b;0<=a&&52>a&&0===a%1&&this.data_[a]!=b&&(this.data_[a]=b,this.h=-1)};
mc.prototype.get=function(a){return!!this.data_[a]};
function nc(a){-1==a.h&&(a.h=Qa(a.data_,function(b,c,d){return c?b+Math.pow(2,d):b},0));
return a.h}
;function oc(a,b){this.j=a;this.l=b;this.i=0;this.h=null}
oc.prototype.get=function(){if(0<this.i){this.i--;var a=this.h;this.h=a.next;a.next=null}else a=this.j();return a};
function pc(a,b){a.l(b);100>a.i&&(a.i++,b.next=a.h,a.h=b)}
;var qc;function rc(){var a=z.MessageChannel;"undefined"===typeof a&&"undefined"!==typeof window&&window.postMessage&&window.addEventListener&&!I("Presto")&&(a=function(){var e=Ub();e.style.display="none";document.documentElement.appendChild(e);var f=e.contentWindow;e=f.document;e.open();e.close();var g="callImmediate"+Math.random(),h="file:"==f.location.protocol?"*":f.location.protocol+"//"+f.location.host;e=Ka(function(k){if(("*"==h||k.origin==h)&&k.data==g)this.port1.onmessage()},this);
f.addEventListener("message",e,!1);this.port1={};this.port2={postMessage:function(){f.postMessage(g,h)}}});
if("undefined"!==typeof a&&!I("Trident")&&!I("MSIE")){var b=new a,c={},d=c;b.port1.onmessage=function(){if(void 0!==c.next){c=c.next;var e=c.ha;c.ha=null;e()}};
return function(e){d.next={ha:e};d=d.next;b.port2.postMessage(0)}}return function(e){z.setTimeout(e,0)}}
;function sc(a){z.setTimeout(function(){throw a;},0)}
;function tc(){this.i=this.h=null}
tc.prototype.add=function(a,b){var c=uc.get();c.set(a,b);this.i?this.i.next=c:this.h=c;this.i=c};
tc.prototype.remove=function(){var a=null;this.h&&(a=this.h,this.h=this.h.next,this.h||(this.i=null),a.next=null);return a};
var uc=new oc(function(){return new vc},function(a){return a.reset()});
function vc(){this.next=this.scope=this.h=null}
vc.prototype.set=function(a,b){this.h=a;this.scope=b;this.next=null};
vc.prototype.reset=function(){this.next=this.scope=this.h=null};function wc(a,b){xc||yc();zc||(xc(),zc=!0);Ac.add(a,b)}
var xc;function yc(){if(z.Promise&&z.Promise.resolve){var a=z.Promise.resolve(void 0);xc=function(){a.then(Bc)}}else xc=function(){var b=Bc;
"function"!==typeof z.setImmediate||z.Window&&z.Window.prototype&&!I("Edge")&&z.Window.prototype.setImmediate==z.setImmediate?(qc||(qc=rc()),qc(b)):z.setImmediate(b)}}
var zc=!1,Ac=new tc;function Bc(){for(var a;a=Ac.remove();){try{a.h.call(a.scope)}catch(b){sc(b)}pc(uc,a)}zc=!1}
;function Cc(a,b){this.h=a[z.Symbol.iterator]();this.i=b;this.j=0}
Cc.prototype[Symbol.iterator]=function(){return this};
Cc.prototype.next=function(){var a=this.h.next();return{value:a.done?void 0:this.i.call(void 0,a.value,this.j++),done:a.done}};
function Dc(a,b){return new Cc(a,b)}
;function Ec(){this.blockSize=-1}
;function Fc(){this.blockSize=-1;this.blockSize=64;this.h=[];this.m=[];this.s=[];this.j=[];this.j[0]=128;for(var a=1;a<this.blockSize;++a)this.j[a]=0;this.l=this.i=0;this.reset()}
G(Fc,Ec);Fc.prototype.reset=function(){this.h[0]=1732584193;this.h[1]=4023233417;this.h[2]=2562383102;this.h[3]=271733878;this.h[4]=3285377520;this.l=this.i=0};
function Gc(a,b,c){c||(c=0);var d=a.s;if("string"===typeof b)for(var e=0;16>e;e++)d[e]=b.charCodeAt(c)<<24|b.charCodeAt(c+1)<<16|b.charCodeAt(c+2)<<8|b.charCodeAt(c+3),c+=4;else for(e=0;16>e;e++)d[e]=b[c]<<24|b[c+1]<<16|b[c+2]<<8|b[c+3],c+=4;for(e=16;80>e;e++){var f=d[e-3]^d[e-8]^d[e-14]^d[e-16];d[e]=(f<<1|f>>>31)&4294967295}b=a.h[0];c=a.h[1];var g=a.h[2],h=a.h[3],k=a.h[4];for(e=0;80>e;e++){if(40>e)if(20>e){f=h^c&(g^h);var l=1518500249}else f=c^g^h,l=1859775393;else 60>e?(f=c&g|h&(c|g),l=2400959708):
(f=c^g^h,l=3395469782);f=(b<<5|b>>>27)+f+k+l+d[e]&4294967295;k=h;h=g;g=(c<<30|c>>>2)&4294967295;c=b;b=f}a.h[0]=a.h[0]+b&4294967295;a.h[1]=a.h[1]+c&4294967295;a.h[2]=a.h[2]+g&4294967295;a.h[3]=a.h[3]+h&4294967295;a.h[4]=a.h[4]+k&4294967295}
Fc.prototype.update=function(a,b){if(null!=a){void 0===b&&(b=a.length);for(var c=b-this.blockSize,d=0,e=this.m,f=this.i;d<b;){if(0==f)for(;d<=c;)Gc(this,a,d),d+=this.blockSize;if("string"===typeof a)for(;d<b;){if(e[f]=a.charCodeAt(d),++f,++d,f==this.blockSize){Gc(this,e);f=0;break}}else for(;d<b;)if(e[f]=a[d],++f,++d,f==this.blockSize){Gc(this,e);f=0;break}}this.i=f;this.l+=b}};
Fc.prototype.digest=function(){var a=[],b=8*this.l;56>this.i?this.update(this.j,56-this.i):this.update(this.j,this.blockSize-(this.i-56));for(var c=this.blockSize-1;56<=c;c--)this.m[c]=b&255,b/=256;Gc(this,this.m);for(c=b=0;5>c;c++)for(var d=24;0<=d;d-=8)a[b]=this.h[c]>>d&255,++b;return a};function Hc(a){var b=B("window.location.href");null==a&&(a='Unknown Error of type "null/undefined"');if("string"===typeof a)return{message:a,name:"Unknown error",lineNumber:"Not available",fileName:b,stack:"Not available"};var c=!1;try{var d=a.lineNumber||a.line||"Not available"}catch(g){d="Not available",c=!0}try{var e=a.fileName||a.filename||a.sourceURL||z.$googDebugFname||b}catch(g){e="Not available",c=!0}b=Ic(a);if(!(!c&&a.lineNumber&&a.fileName&&a.stack&&a.message&&a.name)){c=a.message;if(null==
c){if(a.constructor&&a.constructor instanceof Function){if(a.constructor.name)c=a.constructor.name;else if(c=a.constructor,Jc[c])c=Jc[c];else{c=String(c);if(!Jc[c]){var f=/function\s+([^\(]+)/m.exec(c);Jc[c]=f?f[1]:"[Anonymous]"}c=Jc[c]}c='Unknown Error of type "'+c+'"'}else c="Unknown Error of unknown type";"function"===typeof a.toString&&Object.prototype.toString!==a.toString&&(c+=": "+a.toString())}return{message:c,name:a.name||"UnknownError",lineNumber:d,fileName:e,stack:b||"Not available"}}a.stack=
b;return{message:a.message,name:a.name,lineNumber:a.lineNumber,fileName:a.fileName,stack:a.stack}}
function Ic(a,b){b||(b={});b[Kc(a)]=!0;var c=a.stack||"";(a=a.wa)&&!b[Kc(a)]&&(c+="\nCaused by: ",a.stack&&0==a.stack.indexOf(a.toString())||(c+="string"===typeof a?a:a.message+"\n"),c+=Ic(a,b));return c}
function Kc(a){var b="";"function"===typeof a.toString&&(b=""+a);return b+a.stack}
var Jc={};function Lc(){this.m=this.m;this.s=this.s}
Lc.prototype.m=!1;Lc.prototype.dispose=function(){this.m||(this.m=!0,this.R())};
Lc.prototype.R=function(){if(this.s)for(;this.s.length;)this.s.shift()()};var Mc="StopIteration"in z?z.StopIteration:{message:"StopIteration",stack:""};function Nc(){}
Nc.prototype.next=function(){return Nc.prototype.j.call(this)};
Nc.prototype.j=function(){throw Mc;};
Nc.prototype.C=function(){return this};function Oc(a){if(a instanceof Pc||a instanceof Qc||a instanceof Rc)return a;if("function"==typeof a.next)return new Pc(function(){return Sc(a)});
if("function"==typeof a[Symbol.iterator])return new Pc(function(){return a[Symbol.iterator]()});
if("function"==typeof a.C)return new Pc(function(){return Sc(a.C())});
throw Error("Not an iterator or iterable.");}
function Sc(a){if(!(a instanceof Nc))return a;var b=!1;return{next:function(){for(var c;!b;)try{c=a.next();break}catch(d){if(d!==Mc)throw d;b=!0}return{value:c,done:b}}}}
function Pc(a){this.i=a}
Pc.prototype.C=function(){return new Qc(this.i())};
Pc.prototype[Symbol.iterator]=function(){return new Rc(this.i())};
Pc.prototype.h=function(){return new Rc(this.i())};
function Qc(a){this.i=a}
v(Qc,Nc);Qc.prototype.next=function(){var a=this.i.next();if(a.done)throw Mc;return a.value};
Qc.prototype[Symbol.iterator]=function(){return new Rc(this.i)};
Qc.prototype.h=function(){return new Rc(this.i)};
function Rc(a){Pc.call(this,function(){return a});
this.j=a}
v(Rc,Pc);Rc.prototype.next=function(){return this.j.next()};function Tc(a,b){this.i={};this.h=[];this.j=this.size=0;var c=arguments.length;if(1<c){if(c%2)throw Error("Uneven number of arguments");for(var d=0;d<c;d+=2)this.set(arguments[d],arguments[d+1])}else if(a)if(a instanceof Tc)for(c=Uc(a),d=0;d<c.length;d++)this.set(c[d],a.get(c[d]));else for(d in a)this.set(d,a[d])}
function Uc(a){Vc(a);return a.h.concat()}
n=Tc.prototype;n.has=function(a){return Wc(this.i,a)};
n.equals=function(a,b){if(this===a)return!0;if(this.size!=a.size)return!1;b=b||Xc;Vc(this);for(var c,d=0;c=this.h[d];d++)if(!b(this.get(c),a.get(c)))return!1;return!0};
function Xc(a,b){return a===b}
n.isEmpty=function(){return 0==this.size};
n.clear=function(){this.i={};this.j=this.size=this.h.length=0};
n.remove=function(a){return this.delete(a)};
n.delete=function(a){return Wc(this.i,a)?(delete this.i[a],--this.size,this.j++,this.h.length>2*this.size&&Vc(this),!0):!1};
function Vc(a){if(a.size!=a.h.length){for(var b=0,c=0;b<a.h.length;){var d=a.h[b];Wc(a.i,d)&&(a.h[c++]=d);b++}a.h.length=c}if(a.size!=a.h.length){var e={};for(c=b=0;b<a.h.length;)d=a.h[b],Wc(e,d)||(a.h[c++]=d,e[d]=1),b++;a.h.length=c}}
n.get=function(a,b){return Wc(this.i,a)?this.i[a]:b};
n.set=function(a,b){Wc(this.i,a)||(this.size+=1,this.h.push(a),this.j++);this.i[a]=b};
n.forEach=function(a,b){for(var c=Uc(this),d=0;d<c.length;d++){var e=c[d],f=this.get(e);a.call(b,f,e,this)}};
n.clone=function(){return new Tc(this)};
n.keys=function(){return Oc(this.C(!0)).h()};
n.values=function(){return Oc(this.C(!1)).h()};
n.entries=function(){var a=this;return Dc(this.keys(),function(b){return[b,a.get(b)]})};
n.C=function(a){Vc(this);var b=0,c=this.j,d=this,e=new Nc;e.next=function(){if(c!=d.j)throw Error("The map has changed since the iterator was created");if(b>=d.h.length)throw Mc;var f=d.h[b++];return a?f:d.i[f]};
return e};
function Wc(a,b){return Object.prototype.hasOwnProperty.call(a,b)}
;var Yc=function(){if(!z.addEventListener||!Object.defineProperty)return!1;var a=!1,b=Object.defineProperty({},"passive",{get:function(){a=!0}});
try{z.addEventListener("test",Da,b),z.removeEventListener("test",Da,b)}catch(c){}return a}();function Zc(a,b){this.type=a;this.h=this.target=b;this.defaultPrevented=this.j=!1}
Zc.prototype.stopPropagation=function(){this.j=!0};
Zc.prototype.preventDefault=function(){this.defaultPrevented=!0};function $c(a,b){Zc.call(this,a?a.type:"");this.relatedTarget=this.h=this.target=null;this.button=this.screenY=this.screenX=this.clientY=this.clientX=0;this.key="";this.charCode=this.keyCode=0;this.metaKey=this.shiftKey=this.altKey=this.ctrlKey=!1;this.state=null;this.pointerId=0;this.pointerType="";this.i=null;a&&this.init(a,b)}
G($c,Zc);var ad={2:"touch",3:"pen",4:"mouse"};
$c.prototype.init=function(a,b){var c=this.type=a.type,d=a.changedTouches&&a.changedTouches.length?a.changedTouches[0]:null;this.target=a.target||a.srcElement;this.h=b;if(b=a.relatedTarget){if(Db){a:{try{zb(b.nodeName);var e=!0;break a}catch(f){}e=!1}e||(b=null)}}else"mouseover"==c?b=a.fromElement:"mouseout"==c&&(b=a.toElement);this.relatedTarget=b;d?(this.clientX=void 0!==d.clientX?d.clientX:d.pageX,this.clientY=void 0!==d.clientY?d.clientY:d.pageY,this.screenX=d.screenX||0,this.screenY=d.screenY||
0):(this.clientX=void 0!==a.clientX?a.clientX:a.pageX,this.clientY=void 0!==a.clientY?a.clientY:a.pageY,this.screenX=a.screenX||0,this.screenY=a.screenY||0);this.button=a.button;this.keyCode=a.keyCode||0;this.key=a.key||"";this.charCode=a.charCode||("keypress"==c?a.keyCode:0);this.ctrlKey=a.ctrlKey;this.altKey=a.altKey;this.shiftKey=a.shiftKey;this.metaKey=a.metaKey;this.pointerId=a.pointerId||0;this.pointerType="string"===typeof a.pointerType?a.pointerType:ad[a.pointerType]||"";this.state=a.state;
this.i=a;a.defaultPrevented&&$c.G.preventDefault.call(this)};
$c.prototype.stopPropagation=function(){$c.G.stopPropagation.call(this);this.i.stopPropagation?this.i.stopPropagation():this.i.cancelBubble=!0};
$c.prototype.preventDefault=function(){$c.G.preventDefault.call(this);var a=this.i;a.preventDefault?a.preventDefault():a.returnValue=!1};var bd="closure_listenable_"+(1E6*Math.random()|0);var cd=0;function dd(a,b,c,d,e){this.listener=a;this.h=null;this.src=b;this.type=c;this.capture=!!d;this.aa=e;this.key=++cd;this.S=this.W=!1}
function ed(a){a.S=!0;a.listener=null;a.h=null;a.src=null;a.aa=null}
;function fd(a){this.src=a;this.listeners={};this.h=0}
fd.prototype.add=function(a,b,c,d,e){var f=a.toString();a=this.listeners[f];a||(a=this.listeners[f]=[],this.h++);var g=gd(a,b,d,e);-1<g?(b=a[g],c||(b.W=!1)):(b=new dd(b,this.src,f,!!d,e),b.W=c,a.push(b));return b};
fd.prototype.remove=function(a,b,c,d){a=a.toString();if(!(a in this.listeners))return!1;var e=this.listeners[a];b=gd(e,b,c,d);return-1<b?(ed(e[b]),Array.prototype.splice.call(e,b,1),0==e.length&&(delete this.listeners[a],this.h--),!0):!1};
function hd(a,b){var c=b.type;c in a.listeners&&Sa(a.listeners[c],b)&&(ed(b),0==a.listeners[c].length&&(delete a.listeners[c],a.h--))}
function gd(a,b,c,d){for(var e=0;e<a.length;++e){var f=a[e];if(!f.S&&f.listener==b&&f.capture==!!c&&f.aa==d)return e}return-1}
;var id="closure_lm_"+(1E6*Math.random()|0),jd={},kd=0;function ld(a,b,c,d,e){if(d&&d.once)md(a,b,c,d,e);else if(Array.isArray(b))for(var f=0;f<b.length;f++)ld(a,b[f],c,d,e);else c=nd(c),a&&a[bd]?a.ba(b,c,D(d)?!!d.capture:!!d,e):od(a,b,c,!1,d,e)}
function od(a,b,c,d,e,f){if(!b)throw Error("Invalid event type");var g=D(e)?!!e.capture:!!e,h=pd(a);h||(a[id]=h=new fd(a));c=h.add(b,c,d,g,f);if(!c.h){d=qd();c.h=d;d.src=a;d.listener=c;if(a.addEventListener)Yc||(e=g),void 0===e&&(e=!1),a.addEventListener(b.toString(),d,e);else if(a.attachEvent)a.attachEvent(rd(b.toString()),d);else if(a.addListener&&a.removeListener)a.addListener(d);else throw Error("addEventListener and attachEvent are unavailable.");kd++}}
function qd(){function a(c){return b.call(a.src,a.listener,c)}
var b=sd;return a}
function md(a,b,c,d,e){if(Array.isArray(b))for(var f=0;f<b.length;f++)md(a,b[f],c,d,e);else c=nd(c),a&&a[bd]?a.i.add(String(b),c,!0,D(d)?!!d.capture:!!d,e):od(a,b,c,!0,d,e)}
function td(a,b,c,d,e){if(Array.isArray(b))for(var f=0;f<b.length;f++)td(a,b[f],c,d,e);else(d=D(d)?!!d.capture:!!d,c=nd(c),a&&a[bd])?a.i.remove(String(b),c,d,e):a&&(a=pd(a))&&(b=a.listeners[b.toString()],a=-1,b&&(a=gd(b,c,d,e)),(c=-1<a?b[a]:null)&&ud(c))}
function ud(a){if("number"!==typeof a&&a&&!a.S){var b=a.src;if(b&&b[bd])hd(b.i,a);else{var c=a.type,d=a.h;b.removeEventListener?b.removeEventListener(c,d,a.capture):b.detachEvent?b.detachEvent(rd(c),d):b.addListener&&b.removeListener&&b.removeListener(d);kd--;(c=pd(b))?(hd(c,a),0==c.h&&(c.src=null,b[id]=null)):ed(a)}}}
function rd(a){return a in jd?jd[a]:jd[a]="on"+a}
function sd(a,b){if(a.S)a=!0;else{b=new $c(b,this);var c=a.listener,d=a.aa||a.src;a.W&&ud(a);a=c.call(d,b)}return a}
function pd(a){a=a[id];return a instanceof fd?a:null}
var vd="__closure_events_fn_"+(1E9*Math.random()>>>0);function nd(a){if("function"===typeof a)return a;a[vd]||(a[vd]=function(b){return a.handleEvent(b)});
return a[vd]}
;function L(){Lc.call(this);this.i=new fd(this);this.da=this;this.M=null}
G(L,Lc);L.prototype[bd]=!0;L.prototype.addEventListener=function(a,b,c,d){ld(this,a,b,c,d)};
L.prototype.removeEventListener=function(a,b,c,d){td(this,a,b,c,d)};
function M(a,b){var c=a.M;if(c){var d=[];for(var e=1;c;c=c.M)d.push(c),++e}a=a.da;c=b.type||b;"string"===typeof b?b=new Zc(b,a):b instanceof Zc?b.target=b.target||a:(e=b,b=new Zc(c,a),bb(b,e));e=!0;if(d)for(var f=d.length-1;!b.j&&0<=f;f--){var g=b.h=d[f];e=wd(g,c,!0,b)&&e}b.j||(g=b.h=a,e=wd(g,c,!0,b)&&e,b.j||(e=wd(g,c,!1,b)&&e));if(d)for(f=0;!b.j&&f<d.length;f++)g=b.h=d[f],e=wd(g,c,!1,b)&&e}
L.prototype.R=function(){L.G.R.call(this);if(this.i){var a=this.i,b=0,c;for(c in a.listeners){for(var d=a.listeners[c],e=0;e<d.length;e++)++b,ed(d[e]);delete a.listeners[c];a.h--}}this.M=null};
L.prototype.ba=function(a,b,c,d){return this.i.add(String(a),b,!1,c,d)};
function wd(a,b,c,d){b=a.i.listeners[String(b)];if(!b)return!0;b=b.concat();for(var e=!0,f=0;f<b.length;++f){var g=b[f];if(g&&!g.S&&g.capture==c){var h=g.listener,k=g.aa||g.src;g.W&&hd(a.i,g);e=!1!==h.call(k,d)&&e}}return e&&!d.defaultPrevented}
;var xd=z.JSON.stringify;function O(a){this.h=0;this.o=void 0;this.l=this.i=this.j=null;this.m=this.s=!1;if(a!=Da)try{var b=this;a.call(void 0,function(c){yd(b,2,c)},function(c){yd(b,3,c)})}catch(c){yd(this,3,c)}}
function zd(){this.next=this.context=this.onRejected=this.i=this.h=null;this.j=!1}
zd.prototype.reset=function(){this.context=this.onRejected=this.i=this.h=null;this.j=!1};
var Ad=new oc(function(){return new zd},function(a){a.reset()});
function Bd(a,b,c){var d=Ad.get();d.i=a;d.onRejected=b;d.context=c;return d}
O.prototype.then=function(a,b,c){return Cd(this,"function"===typeof a?a:null,"function"===typeof b?b:null,c)};
O.prototype.$goog_Thenable=!0;O.prototype.cancel=function(a){if(0==this.h){var b=new Dd(a);wc(function(){Ed(this,b)},this)}};
function Ed(a,b){if(0==a.h)if(a.j){var c=a.j;if(c.i){for(var d=0,e=null,f=null,g=c.i;g&&(g.j||(d++,g.h==a&&(e=g),!(e&&1<d)));g=g.next)e||(f=g);e&&(0==c.h&&1==d?Ed(c,b):(f?(d=f,d.next==c.l&&(c.l=d),d.next=d.next.next):Fd(c),Gd(c,e,3,b)))}a.j=null}else yd(a,3,b)}
function Hd(a,b){a.i||2!=a.h&&3!=a.h||Id(a);a.l?a.l.next=b:a.i=b;a.l=b}
function Cd(a,b,c,d){var e=Bd(null,null,null);e.h=new O(function(f,g){e.i=b?function(h){try{var k=b.call(d,h);f(k)}catch(l){g(l)}}:f;
e.onRejected=c?function(h){try{var k=c.call(d,h);void 0===k&&h instanceof Dd?g(h):f(k)}catch(l){g(l)}}:g});
e.h.j=a;Hd(a,e);return e.h}
O.prototype.B=function(a){this.h=0;yd(this,2,a)};
O.prototype.M=function(a){this.h=0;yd(this,3,a)};
function yd(a,b,c){if(0==a.h){a===c&&(b=3,c=new TypeError("Promise cannot resolve to itself"));a.h=1;a:{var d=c,e=a.B,f=a.M;if(d instanceof O){Hd(d,Bd(e||Da,f||null,a));var g=!0}else{if(d)try{var h=!!d.$goog_Thenable}catch(l){h=!1}else h=!1;if(h)d.then(e,f,a),g=!0;else{if(D(d))try{var k=d.then;if("function"===typeof k){Jd(d,k,e,f,a);g=!0;break a}}catch(l){f.call(a,l);g=!0;break a}g=!1}}}g||(a.o=c,a.h=b,a.j=null,Id(a),3!=b||c instanceof Dd||Kd(a,c))}}
function Jd(a,b,c,d,e){function f(k){h||(h=!0,d.call(e,k))}
function g(k){h||(h=!0,c.call(e,k))}
var h=!1;try{b.call(a,g,f)}catch(k){f(k)}}
function Id(a){a.s||(a.s=!0,wc(a.v,a))}
function Fd(a){var b=null;a.i&&(b=a.i,a.i=b.next,b.next=null);a.i||(a.l=null);return b}
O.prototype.v=function(){for(var a;a=Fd(this);)Gd(this,a,this.h,this.o);this.s=!1};
function Gd(a,b,c,d){if(3==c&&b.onRejected&&!b.j)for(;a&&a.m;a=a.j)a.m=!1;if(b.h)b.h.j=null,Ld(b,c,d);else try{b.j?b.i.call(b.context):Ld(b,c,d)}catch(e){Md.call(null,e)}pc(Ad,b)}
function Ld(a,b,c){2==b?a.i.call(a.context,c):a.onRejected&&a.onRejected.call(a.context,c)}
function Kd(a,b){a.m=!0;wc(function(){a.m&&Md.call(null,b)})}
var Md=sc;function Dd(a){Ma.call(this,a)}
G(Dd,Ma);Dd.prototype.name="cancel";function P(a){Lc.call(this);this.o=1;this.j=[];this.l=0;this.h=[];this.i={};this.v=!!a}
G(P,Lc);n=P.prototype;n.subscribe=function(a,b,c){var d=this.i[a];d||(d=this.i[a]=[]);var e=this.o;this.h[e]=a;this.h[e+1]=b;this.h[e+2]=c;this.o=e+3;d.push(e);return e};
function Nd(a,b,c){var d=Od;if(a=d.i[a]){var e=d.h;(a=Ra(a,function(f){return e[f+1]==b&&e[f+2]==c}))&&d.T(a)}}
n.T=function(a){var b=this.h[a];if(b){var c=this.i[b];0!=this.l?(this.j.push(a),this.h[a+1]=Da):(c&&Sa(c,a),delete this.h[a],delete this.h[a+1],delete this.h[a+2])}return!!b};
n.O=function(a,b){var c=this.i[a];if(c){for(var d=Array(arguments.length-1),e=1,f=arguments.length;e<f;e++)d[e-1]=arguments[e];if(this.v)for(e=0;e<c.length;e++){var g=c[e];Pd(this.h[g+1],this.h[g+2],d)}else{this.l++;try{for(e=0,f=c.length;e<f;e++)g=c[e],this.h[g+1].apply(this.h[g+2],d)}finally{if(this.l--,0<this.j.length&&0==this.l)for(;c=this.j.pop();)this.T(c)}}return 0!=e}return!1};
function Pd(a,b,c){wc(function(){a.apply(b,c)})}
n.clear=function(a){if(a){var b=this.i[a];b&&(H(b,this.T,this),delete this.i[a])}else this.h.length=0,this.i={}};
n.R=function(){P.G.R.call(this);this.clear();this.j.length=0};function Qd(a){this.h=a}
Qd.prototype.set=function(a,b){void 0===b?this.h.remove(a):this.h.set(a,xd(b))};
Qd.prototype.get=function(a){try{var b=this.h.get(a)}catch(c){return}if(null!==b)try{return JSON.parse(b)}catch(c){throw"Storage: Invalid value was encountered";}};
Qd.prototype.remove=function(a){this.h.remove(a)};function Rd(a){this.h=a}
G(Rd,Qd);function Sd(a){this.data=a}
function Td(a){return void 0===a||a instanceof Sd?a:new Sd(a)}
Rd.prototype.set=function(a,b){Rd.G.set.call(this,a,Td(b))};
Rd.prototype.i=function(a){a=Rd.G.get.call(this,a);if(void 0===a||a instanceof Object)return a;throw"Storage: Invalid value was encountered";};
Rd.prototype.get=function(a){if(a=this.i(a)){if(a=a.data,void 0===a)throw"Storage: Invalid value was encountered";}else a=void 0;return a};function Ud(a){this.h=a}
G(Ud,Rd);Ud.prototype.set=function(a,b,c){if(b=Td(b)){if(c){if(c<Date.now()){Ud.prototype.remove.call(this,a);return}b.expiration=c}b.creation=Date.now()}Ud.G.set.call(this,a,b)};
Ud.prototype.i=function(a){var b=Ud.G.i.call(this,a);if(b){var c=b.creation,d=b.expiration;if(d&&d<Date.now()||c&&c>Date.now())Ud.prototype.remove.call(this,a);else return b}};function Vd(){}
;function Wd(){}
G(Wd,Vd);Wd.prototype[Symbol.iterator]=function(){return Oc(this.C(!0)).h()};
Wd.prototype.clear=function(){var a=Array.from(this);a=u(a);for(var b=a.next();!b.done;b=a.next())this.remove(b.value)};function Xd(a){this.h=a}
G(Xd,Wd);n=Xd.prototype;n.isAvailable=function(){if(!this.h)return!1;try{return this.h.setItem("__sak","1"),this.h.removeItem("__sak"),!0}catch(a){return!1}};
n.set=function(a,b){try{this.h.setItem(a,b)}catch(c){if(0==this.h.length)throw"Storage mechanism: Storage disabled";throw"Storage mechanism: Quota exceeded";}};
n.get=function(a){a=this.h.getItem(a);if("string"!==typeof a&&null!==a)throw"Storage mechanism: Invalid value was encountered";return a};
n.remove=function(a){this.h.removeItem(a)};
n.C=function(a){var b=0,c=this.h,d=new Nc;d.next=function(){if(b>=c.length)throw Mc;var e=c.key(b++);if(a)return e;e=c.getItem(e);if("string"!==typeof e)throw"Storage mechanism: Invalid value was encountered";return e};
return d};
n.clear=function(){this.h.clear()};
n.key=function(a){return this.h.key(a)};function Yd(){var a=null;try{a=window.localStorage||null}catch(b){}this.h=a}
G(Yd,Xd);function Zd(a,b){this.i=a;this.h=null;if(Bb&&!(9<=Number(Nb))){$d||($d=new Tc);this.h=$d.get(a);this.h||(b?this.h=document.getElementById(b):(this.h=document.createElement("userdata"),this.h.addBehavior("#default#userData"),document.body.appendChild(this.h)),$d.set(a,this.h));try{this.h.load(this.i)}catch(c){this.h=null}}}
G(Zd,Wd);var ae={".":".2E","!":".21","~":".7E","*":".2A","'":".27","(":".28",")":".29","%":"."},$d=null;function be(a){return"_"+encodeURIComponent(a).replace(/[.!~*'()%]/g,function(b){return ae[b]})}
n=Zd.prototype;n.isAvailable=function(){return!!this.h};
n.set=function(a,b){this.h.setAttribute(be(a),b);ce(this)};
n.get=function(a){a=this.h.getAttribute(be(a));if("string"!==typeof a&&null!==a)throw"Storage mechanism: Invalid value was encountered";return a};
n.remove=function(a){this.h.removeAttribute(be(a));ce(this)};
n.C=function(a){var b=0,c=this.h.XMLDocument.documentElement.attributes,d=new Nc;d.next=function(){if(b>=c.length)throw Mc;var e=c[b++];if(a)return decodeURIComponent(e.nodeName.replace(/\./g,"%")).substr(1);e=e.nodeValue;if("string"!==typeof e)throw"Storage mechanism: Invalid value was encountered";return e};
return d};
n.clear=function(){for(var a=this.h.XMLDocument.documentElement,b=a.attributes.length;0<b;b--)a.removeAttribute(a.attributes[b-1].nodeName);ce(this)};
function ce(a){try{a.h.save(a.i)}catch(b){throw"Storage mechanism: Quota exceeded";}}
;function de(a,b){this.i=a;this.h=b+"::"}
G(de,Wd);de.prototype.set=function(a,b){this.i.set(this.h+a,b)};
de.prototype.get=function(a){return this.i.get(this.h+a)};
de.prototype.remove=function(a){this.i.remove(this.h+a)};
de.prototype.C=function(a){var b=this.i.C(!0),c=this,d=new Nc;d.next=function(){for(var e=b.next();e.substr(0,c.h.length)!=c.h;)e=b.next();return a?e.substr(c.h.length):c.i.get(e)};
return d};var ee,fe,ge=z.window,he=(null===(ee=null===ge||void 0===ge?void 0:ge.yt)||void 0===ee?void 0:ee.config_)||(null===(fe=null===ge||void 0===ge?void 0:ge.ytcfg)||void 0===fe?void 0:fe.data_)||{};F("yt.config_",he);function ie(a){for(var b=0;b<arguments.length;++b);b=arguments;1<b.length?he[b[0]]=b[1]:1===b.length&&Object.assign(he,b[0])}
function S(a,b){return a in he?he[a]:b}
;var je=[];function ke(a){je.forEach(function(b){return b(a)})}
function le(a){return a&&window.yterr?function(){try{return a.apply(this,arguments)}catch(b){me(b)}}:a}
function me(a){var b=B("yt.logging.errors.log");b?b(a,"ERROR",void 0,void 0,void 0):(b=S("ERRORS",[]),b.push([a,"ERROR",void 0,void 0,void 0]),ie("ERRORS",b));ke(a)}
function ne(a){var b=B("yt.logging.errors.log");b?b(a,"WARNING",void 0,void 0,void 0):(b=S("ERRORS",[]),b.push([a,"WARNING",void 0,void 0,void 0]),ie("ERRORS",b))}
;var oe=0;F("ytDomDomGetNextId",B("ytDomDomGetNextId")||function(){return++oe});var pe={stopImmediatePropagation:1,stopPropagation:1,preventMouseEvent:1,preventManipulation:1,preventDefault:1,layerX:1,layerY:1,screenX:1,screenY:1,scale:1,rotation:1,webkitMovementX:1,webkitMovementY:1};
function qe(a){this.type="";this.state=this.source=this.data=this.currentTarget=this.relatedTarget=this.target=null;this.charCode=this.keyCode=0;this.metaKey=this.shiftKey=this.ctrlKey=this.altKey=!1;this.rotation=this.clientY=this.clientX=0;this.changedTouches=this.touches=null;try{if(a=a||window.event){this.event=a;for(var b in a)b in pe||(this[b]=a[b]);this.rotation=a.rotation;var c=a.target||a.srcElement;c&&3==c.nodeType&&(c=c.parentNode);this.target=c;var d=a.relatedTarget;if(d)try{d=d.nodeName?
d:null}catch(e){d=null}else"mouseover"==this.type?d=a.fromElement:"mouseout"==this.type&&(d=a.toElement);this.relatedTarget=d;this.clientX=void 0!=a.clientX?a.clientX:a.pageX;this.clientY=void 0!=a.clientY?a.clientY:a.pageY;this.keyCode=a.keyCode?a.keyCode:a.which;this.charCode=a.charCode||("keypress"==this.type?this.keyCode:0);this.altKey=a.altKey;this.ctrlKey=a.ctrlKey;this.shiftKey=a.shiftKey;this.metaKey=a.metaKey}}catch(e){}}
qe.prototype.preventDefault=function(){this.event&&(this.event.returnValue=!1,this.event.preventDefault&&this.event.preventDefault())};
qe.prototype.stopPropagation=function(){this.event&&(this.event.cancelBubble=!0,this.event.stopPropagation&&this.event.stopPropagation())};
qe.prototype.stopImmediatePropagation=function(){this.event&&(this.event.cancelBubble=!0,this.event.stopImmediatePropagation&&this.event.stopImmediatePropagation())};var Ya=z.ytEventsEventsListeners||{};F("ytEventsEventsListeners",Ya);var re=z.ytEventsEventsCounter||{count:0};F("ytEventsEventsCounter",re);
function se(a,b,c,d){d=void 0===d?{}:d;a.addEventListener&&("mouseenter"!=b||"onmouseenter"in document?"mouseleave"!=b||"onmouseenter"in document?"mousewheel"==b&&"MozBoxSizing"in document.documentElement.style&&(b="MozMousePixelScroll"):b="mouseout":b="mouseover");return Xa(function(e){var f="boolean"===typeof e[4]&&e[4]==!!d,g=D(e[4])&&D(d)&&Za(e[4],d);return!!e.length&&e[0]==a&&e[1]==b&&e[2]==c&&(f||g)})}
function te(a){a&&("string"==typeof a&&(a=[a]),H(a,function(b){if(b in Ya){var c=Ya[b],d=c[0],e=c[1],f=c[3];c=c[4];d.removeEventListener?ue()||"boolean"===typeof c?d.removeEventListener(e,f,c):d.removeEventListener(e,f,!!c.capture):d.detachEvent&&d.detachEvent("on"+e,f);delete Ya[b]}}))}
var ue=Oa(function(){var a=!1;try{var b=Object.defineProperty({},"capture",{get:function(){a=!0}});
window.addEventListener("test",null,b)}catch(c){}return a});
function ve(a,b,c){var d=void 0===d?{}:d;if(a&&(a.addEventListener||a.attachEvent)){var e=se(a,b,c,d);if(!e){e=++re.count+"";var f=!("mouseenter"!=b&&"mouseleave"!=b||!a.addEventListener||"onmouseenter"in document);var g=f?function(h){h=new qe(h);if(!Vb(h.relatedTarget,function(k){return k==a}))return h.currentTarget=a,h.type=b,c.call(a,h)}:function(h){h=new qe(h);
h.currentTarget=a;return c.call(a,h)};
g=le(g);a.addEventListener?("mouseenter"==b&&f?b="mouseover":"mouseleave"==b&&f?b="mouseout":"mousewheel"==b&&"MozBoxSizing"in document.documentElement.style&&(b="MozMousePixelScroll"),ue()||"boolean"===typeof d?a.addEventListener(b,g,d):a.addEventListener(b,g,!!d.capture)):a.attachEvent("on"+b,g);Ya[e]=[a,b,c,g,d]}}}
;function we(a,b){"function"===typeof a&&(a=le(a));return window.setTimeout(a,b)}
function xe(a){"function"===typeof a&&(a=le(a));return window.setInterval(a,250)}
;var ye=/^[\w.]*$/,ze={q:!0,search_query:!0};function Ae(a,b){b=a.split(b);for(var c={},d=0,e=b.length;d<e;d++){var f=b[d].split("=");if(1==f.length&&f[0]||2==f.length)try{var g=Be(f[0]||""),h=Be(f[1]||"");g in c?Array.isArray(c[g])?Va(c[g],h):c[g]=[c[g],h]:c[g]=h}catch(p){var k=p,l=f[0],m=String(Ae);k.args=[{key:l,value:f[1],query:a,method:Ce==m?"unchanged":m}];ze.hasOwnProperty(l)||ne(k)}}return c}
var Ce=String(Ae);function De(a){var b=[];Wa(a,function(c,d){var e=encodeURIComponent(String(d)),f;Array.isArray(c)?f=c:f=[c];H(f,function(g){""==g?b.push(e):b.push(e+"="+encodeURIComponent(String(g)))})});
return b.join("&")}
function Ee(a){"?"==a.charAt(0)&&(a=a.substr(1));return Ae(a,"&")}
function Fe(a,b,c){var d=a.split("#",2);a=d[0];d=1<d.length?"#"+d[1]:"";var e=a.split("?",2);a=e[0];e=Ee(e[1]||"");for(var f in b)!c&&null!==e&&f in e||(e[f]=b[f]);b=a;a=vb(e);a?(c=b.indexOf("#"),0>c&&(c=b.length),f=b.indexOf("?"),0>f||f>c?(f=c,e=""):e=b.substring(f+1,c),b=[b.substr(0,f),e,b.substr(c)],c=b[1],b[1]=a?c?c+"&"+a:a:c,a=b[0]+(b[1]?"?"+b[1]:"")+b[2]):a=b;return a+d}
function Ge(a){if(!b)var b=window.location.href;var c=a.match(qb)[1]||null,d=sb(a);c&&d?(a=a.match(qb),b=b.match(qb),a=a[3]==b[3]&&a[1]==b[1]&&a[4]==b[4]):a=d?sb(b)==d&&(Number(b.match(qb)[4]||null)||null)==(Number(a.match(qb)[4]||null)||null):!0;return a}
function Be(a){return a&&a.match(ye)?a:decodeURIComponent(a.replace(/\+/g," "))}
;function T(a){a=He(a);return"string"===typeof a&&"false"===a?!1:!!a}
function Ie(a,b){a=He(a);return void 0===a&&void 0!==b?b:Number(a||0)}
function He(a){var b=S("EXPERIMENTS_FORCED_FLAGS",{});return void 0!==b[a]?b[a]:S("EXPERIMENT_FLAGS",{})[a]}
;function Je(){}
function Ke(a,b){return Le(a,0,b)}
function Me(a,b){return Le(a,1,b)}
;function Ne(){Je.apply(this,arguments)}
v(Ne,Je);function Le(a,b,c){void 0!==c&&Number.isNaN(Number(c))&&(c=void 0);var d=B("yt.scheduler.instance.addJob");return d?d(a,b,c):void 0===c?(a(),NaN):we(a,c||0)}
function Oe(a){if(void 0===a||!Number.isNaN(Number(a))){var b=B("yt.scheduler.instance.cancelJob");b?b(a):window.clearTimeout(a)}}
Ne.prototype.start=function(){var a=B("yt.scheduler.instance.start");a&&a()};Ne.h||(Ne.h=new Ne);function Pe(a){var b=Qe;a=void 0===a?B("yt.ads.biscotti.lastId_")||"":a;var c=Object,d=c.assign,e={};e.dt=ac;e.flash="0";a:{try{var f=b.h.top.location.href}catch(xa){f=2;break a}f=f?f===b.i.location.href?0:1:2}e=(e.frm=f,e);e.u_tz=-(new Date).getTimezoneOffset();var g=void 0===g?K:g;try{var h=g.history.length}catch(xa){h=0}e.u_his=h;e.u_java=!!K.navigator&&"unknown"!==typeof K.navigator.javaEnabled&&!!K.navigator.javaEnabled&&K.navigator.javaEnabled();K.screen&&(e.u_h=K.screen.height,e.u_w=K.screen.width,
e.u_ah=K.screen.availHeight,e.u_aw=K.screen.availWidth,e.u_cd=K.screen.colorDepth);K.navigator&&K.navigator.plugins&&(e.u_nplug=K.navigator.plugins.length);K.navigator&&K.navigator.mimeTypes&&(e.u_nmime=K.navigator.mimeTypes.length);h=b.h;try{var k=h.screenX;var l=h.screenY}catch(xa){}try{var m=h.outerWidth;var p=h.outerHeight}catch(xa){}try{var r=h.innerWidth;var q=h.innerHeight}catch(xa){}try{var y=h.screenLeft;var C=h.screenTop}catch(xa){}try{r=h.innerWidth,q=h.innerHeight}catch(xa){}try{var A=
h.screen.availWidth;var Q=h.screen.availTop}catch(xa){}k=[y,C,k,l,A,Q,m,p,r,q];l=b.h.top;try{var R=(l||window).document,E="CSS1Compat"==R.compatMode?R.documentElement:R.body;var N=(new Tb(E.clientWidth,E.clientHeight)).round()}catch(xa){N=new Tb(-12245933,-12245933)}R=N;N={};E=new mc;z.SVGElement&&z.document.createElementNS&&E.set(0);l=$b();l["allow-top-navigation-by-user-activation"]&&E.set(1);l["allow-popups-to-escape-sandbox"]&&E.set(2);z.crypto&&z.crypto.subtle&&E.set(3);z.TextDecoder&&z.TextEncoder&&
E.set(4);E=nc(E);N.bc=E;N.bih=R.height;N.biw=R.width;N.brdim=k.join();b=b.i;b=(N.vis={visible:1,hidden:2,prerender:3,preview:4,unloaded:5}[b.visibilityState||b.webkitVisibilityState||b.mozVisibilityState||""]||0,N.wgl=!!K.WebGLRenderingContext,N);c=d.call(c,e,b);c.ca_type="image";a&&(c.bid=a);return c}
var Qe=new function(){var a=window.document;this.h=window;this.i=a};
F("yt.ads_.signals_.getAdSignalsString",function(a){return De(Pe(a))});var Re="XMLHttpRequest"in z?function(){return new XMLHttpRequest}:null;
function Se(){if(!Re)return null;var a=Re();return"open"in a?a:null}
;var Te={Authorization:"AUTHORIZATION","X-Goog-Visitor-Id":"SANDBOXED_VISITOR_ID","X-Youtube-Chrome-Connected":"CHROME_CONNECTED_HEADER","X-YouTube-Client-Name":"INNERTUBE_CONTEXT_CLIENT_NAME","X-YouTube-Client-Version":"INNERTUBE_CONTEXT_CLIENT_VERSION","X-YouTube-Delegation-Context":"INNERTUBE_CONTEXT_SERIALIZED_DELEGATION_CONTEXT","X-YouTube-Device":"DEVICE","X-Youtube-Identity-Token":"ID_TOKEN","X-YouTube-Page-CL":"PAGE_CL","X-YouTube-Page-Label":"PAGE_BUILD_LABEL","X-YouTube-Variants-Checksum":"VARIANTS_CHECKSUM"},
Ue="app debugcss debugjs expflag force_ad_params force_ad_encrypted force_viral_ad_response_params forced_experiments innertube_snapshots innertube_goldens internalcountrycode internalipoverride absolute_experiments conditional_experiments sbb sr_bns_address client_dev_root_url".split(" "),Ve=!1;
function We(a,b){b=void 0===b?{}:b;var c=Ge(a),d=T("web_ajax_ignore_global_headers_if_set"),e;for(e in Te){var f=S(Te[e]);!f||!c&&sb(a)||d&&void 0!==b[e]||(b[e]=f)}if(c||!sb(a))b["X-YouTube-Utc-Offset"]=String(-(new Date).getTimezoneOffset());if(c||!sb(a)){try{var g=(new Intl.DateTimeFormat).resolvedOptions().timeZone}catch(h){}g&&(b["X-YouTube-Time-Zone"]=g)}if(c||!sb(a))b["X-YouTube-Ad-Signals"]=De(Pe(void 0));return b}
function Xe(a){var b=window.location.search,c=sb(a);T("debug_handle_relative_url_for_query_forward_killswitch")||c||!Ge(a)||(c=document.location.hostname);var d=rb(a.match(qb)[5]||null);d=(c=c&&(c.endsWith("youtube.com")||c.endsWith("youtube-nocookie.com")))&&d&&d.startsWith("/api/");if(!c||d)return a;var e=Ee(b),f={};H(Ue,function(g){e[g]&&(f[g]=e[g])});
return Fe(a,f||{},!1)}
function Ye(a,b){var c=b.format||"JSON";a=Ze(a,b);var d=$e(a,b),e=!1,f=af(a,function(k){if(!e){e=!0;h&&window.clearTimeout(h);a:switch(k&&"status"in k?k.status:-1){case 200:case 201:case 202:case 203:case 204:case 205:case 206:case 304:var l=!0;break a;default:l=!1}var m=null,p=400<=k.status&&500>k.status,r=500<=k.status&&600>k.status;if(l||p||r)m=bf(a,c,k,b.convertToSafeHtml);if(l)a:if(k&&204==k.status)l=!0;else{switch(c){case "XML":l=0==parseInt(m&&m.return_code,10);break a;case "RAW":l=!0;break a}l=
!!m}m=m||{};p=b.context||z;l?b.onSuccess&&b.onSuccess.call(p,k,m):b.onError&&b.onError.call(p,k,m);b.onFinish&&b.onFinish.call(p,k,m)}},b.method,d,b.headers,b.responseType,b.withCredentials);
if(b.onTimeout&&0<b.timeout){var g=b.onTimeout;var h=we(function(){e||(e=!0,f.abort(),window.clearTimeout(h),g.call(b.context||z,f))},b.timeout)}}
function Ze(a,b){b.includeDomain&&(a=document.location.protocol+"//"+document.location.hostname+(document.location.port?":"+document.location.port:"")+a);var c=S("XSRF_FIELD_NAME",void 0);if(b=b.urlParams)b[c]&&delete b[c],a=Fe(a,b||{},!0);return a}
function $e(a,b){var c=S("XSRF_FIELD_NAME",void 0),d=S("XSRF_TOKEN",void 0),e=b.postBody||"",f=b.postParams,g=S("XSRF_FIELD_NAME",void 0),h;b.headers&&(h=b.headers["Content-Type"]);b.excludeXsrf||sb(a)&&!b.withCredentials&&sb(a)!=document.location.hostname||"POST"!=b.method||h&&"application/x-www-form-urlencoded"!=h||b.postParams&&b.postParams[g]||(f||(f={}),f[c]=d);f&&"string"===typeof e&&(e=Ee(e),bb(e,f),e=b.postBodyFormat&&"JSON"==b.postBodyFormat?JSON.stringify(e):vb(e));if(!(a=e)&&(a=f)){a:{for(var k in f){f=
!1;break a}f=!0}a=!f}!Ve&&a&&"POST"!=b.method&&(Ve=!0,me(Error("AJAX request with postData should use POST")));return e}
function bf(a,b,c,d){var e=null;switch(b){case "JSON":try{var f=c.responseText}catch(g){throw d=Error("Error reading responseText"),d.params=a,ne(d),g;}a=c.getResponseHeader("Content-Type")||"";f&&0<=a.indexOf("json")&&(")]}'\n"===f.substring(0,5)&&(f=f.substring(5)),e=JSON.parse(f));break;case "XML":if(a=(a=c.responseXML)?cf(a):null)e={},H(a.getElementsByTagName("*"),function(g){e[g.tagName]=df(g)})}d&&ef(e);
return e}
function ef(a){if(D(a))for(var b in a){var c;(c="html_content"==b)||(c=b.length-5,c=0<=c&&b.indexOf("_html",c)==c);if(c){c=b;var d=a[b];if(void 0===cb){var e=null;var f=z.trustedTypes;if(f&&f.createPolicy){try{e=f.createPolicy("goog#html",{createHTML:La,createScript:La,createScriptURL:La})}catch(g){z.console&&z.console.error(g.message)}cb=e}else cb=e}d=(e=cb)?e.createHTML(d):d;a[c]=new ob(d)}else ef(a[b])}}
function cf(a){return a?(a=("responseXML"in a?a.responseXML:a).getElementsByTagName("root"))&&0<a.length?a[0]:null:null}
function df(a){var b="";H(a.childNodes,function(c){b+=c.nodeValue});
return b}
function af(a,b,c,d,e,f,g){function h(){4==(k&&"readyState"in k?k.readyState:0)&&b&&le(b)(k)}
c=void 0===c?"GET":c;d=void 0===d?"":d;var k=Se();if(!k)return null;"onloadend"in k?k.addEventListener("loadend",h,!1):k.onreadystatechange=h;T("debug_forward_web_query_parameters")&&(a=Xe(a));k.open(c,a,!0);f&&(k.responseType=f);g&&(k.withCredentials=!0);c="POST"==c&&(void 0===window.FormData||!(d instanceof FormData));if(e=We(a,e))for(var l in e)k.setRequestHeader(l,e[l]),"content-type"==l.toLowerCase()&&(c=!1);c&&k.setRequestHeader("Content-Type","application/x-www-form-urlencoded");k.send(d);
return k}
;var ff=Ob||Pb;var gf={},hf=0;function jf(a,b,c){c=void 0===c?"":c;if(kf(a,c))b&&b();else if(c=void 0===c?"":c,a)if(c)af(a,b,"POST",c,void 0);else if(S("USE_NET_AJAX_FOR_PING_TRANSPORT",!1))af(a,b,"GET","",void 0);else{b:{try{var d=new Na({url:a});if(d.j&&d.i||d.l){var e=rb(a.match(qb)[5]||null);var f=!(!e||!e.endsWith("/aclk")||"1"!==xb(a,"ri"));break b}}catch(g){}f=!1}f?kf(a)?(b&&b(),f=!0):f=!1:f=!1;f||lf(a,b)}}
function kf(a,b){try{if(window.navigator&&window.navigator.sendBeacon&&window.navigator.sendBeacon(a,void 0===b?"":b))return!0}catch(c){}return!1}
function lf(a,b){var c=new Image,d=""+hf++;gf[d]=c;c.onload=c.onerror=function(){b&&gf[d]&&b();delete gf[d]};
c.src=a}
;var mf=z.ytPubsubPubsubInstance||new P,nf=z.ytPubsubPubsubSubscribedKeys||{},of=z.ytPubsubPubsubTopicToKeys||{},pf=z.ytPubsubPubsubIsSynchronous||{};P.prototype.subscribe=P.prototype.subscribe;P.prototype.unsubscribeByKey=P.prototype.T;P.prototype.publish=P.prototype.O;P.prototype.clear=P.prototype.clear;F("ytPubsubPubsubInstance",mf);F("ytPubsubPubsubTopicToKeys",of);F("ytPubsubPubsubIsSynchronous",pf);F("ytPubsubPubsubSubscribedKeys",nf);var qf=window,U=qf.ytcsi&&qf.ytcsi.now?qf.ytcsi.now:qf.performance&&qf.performance.timing&&qf.performance.now&&qf.performance.timing.navigationStart?function(){return qf.performance.timing.navigationStart+qf.performance.now()}:function(){return(new Date).getTime()};var rf=Ie("initial_gel_batch_timeout",2E3),sf=Math.pow(2,16)-1,tf=null,uf=0,vf=void 0,wf=0,xf=0,yf=0,zf=!0,Af=z.ytLoggingTransportGELQueue_||new Map;F("ytLoggingTransportGELQueue_",Af);var Bf=z.ytLoggingTransportTokensToCttTargetIds_||{};F("ytLoggingTransportTokensToCttTargetIds_",Bf);
function Cf(a,b){if("log_event"===a.endpoint){var c="";a.Y?c="visitorOnlyApprovedKey":a.H&&(Bf[a.H.token]=Df(a.H),c=a.H.token);var d=Af.get(c)||[];Af.set(c,d);d.push(a.payload);b&&(vf=new b);a=Ie("tvhtml5_logging_max_batch")||Ie("web_logging_max_batch")||100;b=U();d.length>=a?Ef({writeThenSend:!0}):10<=b-yf&&(Ff(),yf=b)}}
function Gf(a,b){if("log_event"===a.endpoint){var c="";a.Y?c="visitorOnlyApprovedKey":a.H&&(Bf[a.H.token]=Df(a.H),c=a.H.token);var d=new Map;d.set(c,[a.payload]);b&&(vf=new b);return new O(function(e){vf&&vf.isReady()?Hf(d,e,{bypassNetworkless:!0}):e()})}}
function Ef(a){a=void 0===a?{}:a;new O(function(b){window.clearTimeout(wf);window.clearTimeout(xf);xf=0;vf&&vf.isReady()?(Hf(Af,b,a),Af.clear()):(Ff(),b())})}
function Ff(){T("web_gel_timeout_cap")&&!xf&&(xf=we(function(){Ef({writeThenSend:!0})},6E4));
window.clearTimeout(wf);var a=S("LOGGING_BATCH_TIMEOUT",Ie("web_gel_debounce_ms",1E4));T("shorten_initial_gel_batch_timeout")&&zf&&(a=rf);wf=we(function(){Ef({writeThenSend:!0})},a)}
function Hf(a,b,c){var d=vf;c=void 0===c?{}:c;var e=Math.round(U()),f=a.size;a=u(a);for(var g=a.next();!g.done;g=a.next()){var h=u(g.value);g=h.next().value;var k=h.next().value;h=$a({context:If(d.config_||Jf())});h.events=k;(k=Bf[g])&&Kf(h,g,k);delete Bf[g];g="visitorOnlyApprovedKey"===g;Lf(h,e,g);T("send_beacon_before_gel")&&window.navigator&&window.navigator.sendBeacon&&!c.writeThenSend&&jf("/generate_204");Mf(d,"log_event",h,{retry:!0,onSuccess:function(){f--;f||b();uf=Math.round(U()-e)},
onError:function(){f--;f||b()},
na:c,Y:g});zf=!1}}
function Lf(a,b,c){a.requestTimeMs=String(b);T("unsplit_gel_payloads_in_logs")&&(a.unsplitGelPayloadsInLogs=!0);!c&&(b=S("EVENT_ID",void 0))&&((c=S("BATCH_CLIENT_COUNTER",void 0)||0)||(c=Math.floor(Math.random()*sf/2)),c++,c>sf&&(c=1),ie("BATCH_CLIENT_COUNTER",c),b={serializedEventId:b,clientCounter:String(c)},a.serializedClientEventId=b,tf&&uf&&T("log_gel_rtt_web")&&(a.previousBatchInfo={serializedClientEventId:tf,roundtripMs:String(uf)}),tf=b,uf=0)}
function Kf(a,b,c){if(c.videoId)var d="VIDEO";else if(c.playlistId)d="PLAYLIST";else return;a.credentialTransferTokenTargetId=c;a.context=a.context||{};a.context.user=a.context.user||{};a.context.user.credentialTransferTokens=[{token:b,scope:d}]}
function Df(a){var b={};a.videoId?b.videoId=a.videoId:a.playlistId&&(b.playlistId=a.playlistId);return b}
;var Nf=z.ytLoggingGelSequenceIdObj_||{};F("ytLoggingGelSequenceIdObj_",Nf);function Of(){if(!z.matchMedia)return"WEB_DISPLAY_MODE_UNKNOWN";try{return z.matchMedia("(display-mode: standalone)").matches?"WEB_DISPLAY_MODE_STANDALONE":z.matchMedia("(display-mode: minimal-ui)").matches?"WEB_DISPLAY_MODE_MINIMAL_UI":z.matchMedia("(display-mode: fullscreen)").matches?"WEB_DISPLAY_MODE_FULLSCREEN":z.matchMedia("(display-mode: browser)").matches?"WEB_DISPLAY_MODE_BROWSER":"WEB_DISPLAY_MODE_UNKNOWN"}catch(a){return"WEB_DISPLAY_MODE_UNKNOWN"}}
;F("ytglobal.prefsUserPrefsPrefs_",B("ytglobal.prefsUserPrefsPrefs_")||{});var Pf={bluetooth:"CONN_DISCO",cellular:"CONN_CELLULAR_UNKNOWN",ethernet:"CONN_WIFI",none:"CONN_NONE",wifi:"CONN_WIFI",wimax:"CONN_CELLULAR_4G",other:"CONN_UNKNOWN",unknown:"CONN_UNKNOWN","slow-2g":"CONN_CELLULAR_2G","2g":"CONN_CELLULAR_2G","3g":"CONN_CELLULAR_3G","4g":"CONN_CELLULAR_4G"},Qf={"slow-2g":"EFFECTIVE_CONNECTION_TYPE_SLOW_2G","2g":"EFFECTIVE_CONNECTION_TYPE_2G","3g":"EFFECTIVE_CONNECTION_TYPE_3G","4g":"EFFECTIVE_CONNECTION_TYPE_4G"};
function Rf(){var a=z.navigator;return a?a.connection:void 0}
;function Sf(){return"INNERTUBE_API_KEY"in he&&"INNERTUBE_API_VERSION"in he}
function Jf(){return{innertubeApiKey:S("INNERTUBE_API_KEY",void 0),innertubeApiVersion:S("INNERTUBE_API_VERSION",void 0),Aa:S("INNERTUBE_CONTEXT_CLIENT_CONFIG_INFO"),Ba:S("INNERTUBE_CONTEXT_CLIENT_NAME","WEB"),innertubeContextClientVersion:S("INNERTUBE_CONTEXT_CLIENT_VERSION",void 0),Da:S("INNERTUBE_CONTEXT_HL",void 0),Ca:S("INNERTUBE_CONTEXT_GL",void 0),Ea:S("INNERTUBE_HOST_OVERRIDE",void 0)||"",Ga:!!S("INNERTUBE_USE_THIRD_PARTY_AUTH",!1),Fa:!!S("INNERTUBE_OMIT_API_KEY_WHEN_AUTH_HEADER_IS_PRESENT",
!1),appInstallData:S("SERIALIZED_CLIENT_CONFIG_DATA",void 0)}}
function If(a){var b={client:{hl:a.Da,gl:a.Ca,clientName:a.Ba,clientVersion:a.innertubeContextClientVersion,configInfo:a.Aa}};T("web_include_ua_it_context")&&navigator.userAgent&&(b.client.userAgent=String(navigator.userAgent));var c=z.devicePixelRatio;c&&1!=c&&(b.client.screenDensityFloat=String(c));c=S("EXPERIMENTS_TOKEN","");""!==c&&(b.client.experimentsToken=c);c=[];var d=S("EXPERIMENTS_FORCED_FLAGS",{});for(e in d)c.push({key:e,value:String(d[e])});var e=S("EXPERIMENT_FLAGS",{});for(var f in e)f.startsWith("force_")&&
void 0===d[f]&&c.push({key:f,value:String(e[f])});0<c.length&&(b.request={internalExperimentFlags:c});f=b.client.clientName;if("WEB"===f||"MWEB"===f||1===f||2===f){if(!T("web_include_display_mode_killswitch")){var g;b.client.mainAppWebInfo=null!=(g=b.client.mainAppWebInfo)?g:{};b.client.mainAppWebInfo.webDisplayMode=Of()}}else if(g=b.client.clientName,("WEB_REMIX"===g||76===g)&&!T("music_web_display_mode_killswitch")){var h;b.client.ma=null!=(h=b.client.ma)?h:{};b.client.ma.webDisplayMode=Of()}a.appInstallData&&
(b.client.configInfo=b.client.configInfo||{},b.client.configInfo.appInstallData=a.appInstallData);S("DELEGATED_SESSION_ID")&&!T("pageid_as_header_web")&&(b.user={onBehalfOfUser:S("DELEGATED_SESSION_ID")});a:{if(h=Rf()){a=Pf[h.type||"unknown"]||"CONN_UNKNOWN";h=Pf[h.effectiveType||"unknown"]||"CONN_UNKNOWN";"CONN_CELLULAR_UNKNOWN"===a&&"CONN_UNKNOWN"!==h&&(a=h);if("CONN_UNKNOWN"!==a)break a;if("CONN_UNKNOWN"!==h){a=h;break a}}a=void 0}a&&(b.client.connectionType=a);T("web_log_effective_connection_type")&&
(a=Rf(),a=null!==a&&void 0!==a&&a.effectiveType?Qf.hasOwnProperty(a.effectiveType)?Qf[a.effectiveType]:"EFFECTIVE_CONNECTION_TYPE_UNKNOWN":void 0,a&&(b.client.effectiveConnectionType=a));a=Object;h=a.assign;g=b.client;f={};e=u(Object.entries(Ee(S("DEVICE",""))));for(c=e.next();!c.done;c=e.next())d=u(c.value),c=d.next().value,d=d.next().value,"cbrand"===c?f.deviceMake=d:"cmodel"===c?f.deviceModel=d:"cbr"===c?f.browserName=d:"cbrver"===c?f.browserVersion=d:"cos"===c?f.osName=d:"cosver"===c?f.osVersion=
d:"cplatform"===c&&(f.platform=d);b.client=h.call(a,g,f);return b}
function Tf(a,b,c){c=void 0===c?{}:c;var d={"X-Goog-Visitor-Id":c.visitorData||S("VISITOR_DATA","")};if(b&&b.includes("www.youtube-nocookie.com"))return d;(b=c.Xa||S("AUTHORIZATION"))||(a?b="Bearer "+B("gapi.auth.getToken")().Wa:b=lc([]));b&&(d.Authorization=b,d["X-Goog-AuthUser"]=S("SESSION_INDEX",0),T("pageid_as_header_web")&&(d["X-Goog-PageId"]=S("DELEGATED_SESSION_ID")));return d}
;function Uf(a){a=Object.assign({},a);delete a.Authorization;var b=lc();if(b){var c=new Fc;c.update(S("INNERTUBE_API_KEY",void 0));c.update(b);b=c.digest();c=3;Ea(b);void 0===c&&(c=0);if(!Sb){Sb={};for(var d="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789".split(""),e=["+/=","+/","-_=","-_.","-_"],f=0;5>f;f++){var g=d.concat(e[f].split(""));Rb[f]=g;for(var h=0;h<g.length;h++){var k=g[h];void 0===Sb[k]&&(Sb[k]=h)}}}c=Rb[c];d=[];for(e=0;e<b.length;e+=3){var l=b[e],m=(f=e+1<b.length)?
b[e+1]:0;k=(g=e+2<b.length)?b[e+2]:0;h=l>>2;l=(l&3)<<4|m>>4;m=(m&15)<<2|k>>6;k&=63;g||(k=64,f||(m=64));d.push(c[h],c[l],c[m]||"",c[k]||"")}a.hash=d.join("")}return a}
;function Vf(a){var b=new Yd;(b=b.isAvailable()?a?new de(b,a):b:null)||(a=new Zd(a||"UserDataSharedStore"),b=a.isAvailable()?a:null);this.h=(a=b)?new Ud(a):null;this.i=document.domain||window.location.hostname}
Vf.prototype.set=function(a,b,c,d){c=c||31104E3;this.remove(a);if(this.h)try{this.h.set(a,b,Date.now()+1E3*c);return}catch(f){}var e="";if(d)try{e=escape(xd(b))}catch(f){return}else e=escape(b);b=this.i;ic.set(""+a,e,{ea:c,path:"/",domain:void 0===b?"youtube.com":b,secure:!1})};
Vf.prototype.get=function(a,b){var c=void 0,d=!this.h;if(!d)try{c=this.h.get(a)}catch(e){d=!0}if(d&&(c=ic.get(""+a,void 0))&&(c=unescape(c),b))try{c=JSON.parse(c)}catch(e){this.remove(a),c=void 0}return c};
Vf.prototype.remove=function(a){this.h&&this.h.remove(a);var b=this.i;ic.remove(""+a,"/",void 0===b?"youtube.com":b)};var Wf;function Xf(){Wf||(Wf=new Vf("yt.innertube"));return Wf}
function Yf(a,b,c,d){if(d)return null;d=Xf().get("nextId",!0)||1;var e=Xf().get("requests",!0)||{};e[d]={method:a,request:b,authState:Uf(c),requestTime:Math.round(U())};Xf().set("nextId",d+1,86400,!0);Xf().set("requests",e,86400,!0);return d}
function Zf(a){var b=Xf().get("requests",!0)||{};delete b[a];Xf().set("requests",b,86400,!0)}
function $f(a){var b=Xf().get("requests",!0);if(b){for(var c in b){var d=b[c];if(!(6E4>Math.round(U())-d.requestTime)){var e=d.authState,f=Uf(Tf(!1));Za(e,f)&&(e=d.request,"requestTimeMs"in e&&(e.requestTimeMs=Math.round(U())),Mf(a,d.method,e,{}));delete b[c]}}Xf().set("requests",b,86400,!0)}}
;var ag=B("ytPubsub2Pubsub2Instance")||new P;P.prototype.subscribe=P.prototype.subscribe;P.prototype.unsubscribeByKey=P.prototype.T;P.prototype.publish=P.prototype.O;P.prototype.clear=P.prototype.clear;F("ytPubsub2Pubsub2Instance",ag);F("ytPubsub2Pubsub2SubscribedKeys",B("ytPubsub2Pubsub2SubscribedKeys")||{});F("ytPubsub2Pubsub2TopicToKeys",B("ytPubsub2Pubsub2TopicToKeys")||{});F("ytPubsub2Pubsub2IsAsync",B("ytPubsub2Pubsub2IsAsync")||{});F("ytPubsub2Pubsub2SkipSubKey",null);function bg(){}
;var cg=function(){var a;return function(){a||(a=new Vf("ytidb"));return a}}();
function dg(){var a;return null===(a=cg())||void 0===a?void 0:a.get("LAST_RESULT_ENTRY_KEY",!0)}
function eg(a){this.h=void 0===a?!1:a;(a=dg())||(a={hasSucceededOnce:this.h});this.i=a;var b,c;T("ytidb_analyze_is_supported")&&(null===(c=cg())||void 0===c?0:c.h)&&(c={hasSucceededOnce:this.i.hasSucceededOnce||this.h},null===(b=cg())||void 0===b?void 0:b.set("LAST_RESULT_ENTRY_KEY",c,2592E3,!0))}
eg.prototype.isSupported=function(){return this.h};var fg=[],gg=!1;function hg(a){gg||(fg.push({type:"ERROR",payload:a}),10<fg.length&&fg.shift())}
function ig(a,b){gg||(fg.push({type:"EVENT",eventType:a,payload:b}),10<fg.length&&fg.shift())}
;function jg(a,b){for(var c=[],d=1;d<arguments.length;++d)c[d-1]=arguments[d];d=Error.call(this,a);this.message=d.message;"stack"in d&&(this.stack=d.stack);d=[];var e=d.concat;if(!(c instanceof Array)){c=u(c);for(var f,g=[];!(f=c.next()).done;)g.push(f.value);c=g}this.args=e.call(d,c)}
v(jg,Error);function kg(){if(void 0!==S("DATASYNC_ID",void 0))return S("DATASYNC_ID",void 0);throw new jg("Datasync ID not set","unknown");}
;function lg(a){if(0<=a.indexOf(":"))throw Error("Database name cannot contain ':'");}
function mg(a){return a.substr(0,a.indexOf(":"))||a}
;var ng={},og=(ng.AUTH_INVALID="No user identifier specified.",ng.EXPLICIT_ABORT="Transaction was explicitly aborted.",ng.IDB_NOT_SUPPORTED="IndexedDB is not supported.",ng.MISSING_OBJECT_STORE="Object store not created.",ng.UNKNOWN_ABORT="Transaction was aborted for unknown reasons.",ng.QUOTA_EXCEEDED="The current transaction exceeded its quota limitations.",ng.QUOTA_MAYBE_EXCEEDED="The current transaction may have failed because of exceeding quota limitations.",ng.EXECUTE_TRANSACTION_ON_CLOSED_DB=
"Can't start a transaction on a closed database",ng),pg={},qg=(pg.AUTH_INVALID="ERROR",pg.EXECUTE_TRANSACTION_ON_CLOSED_DB="WARNING",pg.EXPLICIT_ABORT="IGNORED",pg.IDB_NOT_SUPPORTED="ERROR",pg.MISSING_OBJECT_STORE="ERROR",pg.QUOTA_EXCEEDED="WARNING",pg.QUOTA_MAYBE_EXCEEDED="WARNING",pg.UNKNOWN_ABORT="WARNING",pg),rg={},sg=(rg.AUTH_INVALID=!1,rg.EXECUTE_TRANSACTION_ON_CLOSED_DB=!1,rg.EXPLICIT_ABORT=!1,rg.IDB_NOT_SUPPORTED=!1,rg.MISSING_OBJECT_STORE=!1,rg.QUOTA_EXCEEDED=!1,rg.QUOTA_MAYBE_EXCEEDED=!0,
rg.UNKNOWN_ABORT=!0,rg);function V(a,b,c,d,e){b=void 0===b?{}:b;c=void 0===c?og[a]:c;d=void 0===d?qg[a]:d;e=void 0===e?sg[a]:e;jg.call(this,c,Object.assign({name:"YtIdbKnownError",isSw:void 0===self.document,isIframe:self!==self.top,type:a},b));this.type=a;this.message=c;this.level=d;this.h=e;Object.setPrototypeOf(this,V.prototype)}
v(V,jg);function tg(a){V.call(this,"MISSING_OBJECT_STORE",{bb:a},og.MISSING_OBJECT_STORE);Object.setPrototypeOf(this,tg.prototype)}
v(tg,V);var ug=["The database connection is closing","Can't start a transaction on a closed database","A mutation operation was attempted on a database that did not allow mutations"];
function vg(a,b,c){b=mg(b);var d=a instanceof Error?a:Error("Unexpected error: "+a);if(d instanceof V)return d;if("QuotaExceededError"===d.name)return new V("QUOTA_EXCEEDED",{objectStoreNames:c,dbName:b});if(Qb&&"UnknownError"===d.name)return new V("QUOTA_MAYBE_EXCEEDED",{objectStoreNames:c,dbName:b});if("InvalidStateError"===d.name&&ug.some(function(e){return d.message.includes(e)}))return new V("EXECUTE_TRANSACTION_ON_CLOSED_DB",{objectStoreNames:c,
dbName:b});if("AbortError"===d.name)return new V("UNKNOWN_ABORT",{objectStoreNames:c,dbName:b},d.message);d.args=[{name:"IdbError",cb:d.name,dbName:b,objectStoreNames:c}];d.level="WARNING";return d}
;function wg(a){if(!a)throw Error();throw a;}
function xg(a){return a}
function W(a){function b(e){if("PENDING"===d.state.status){d.state={status:"REJECTED",reason:e};e=u(d.onRejected);for(var f=e.next();!f.done;f=e.next())f=f.value,f()}}
function c(e){if("PENDING"===d.state.status){d.state={status:"FULFILLED",value:e};e=u(d.h);for(var f=e.next();!f.done;f=e.next())f=f.value,f()}}
var d=this;this.i=a;this.state={status:"PENDING"};this.h=[];this.onRejected=[];try{this.i(c,b)}catch(e){b(e)}}
W.all=function(a){return new W(function(b,c){var d=[],e=a.length;0===e&&b(d);for(var f={N:0};f.N<a.length;f={N:f.N},++f.N)yg(W.resolve(a[f.N]).then(function(g){return function(h){d[g.N]=h;e--;0===e&&b(d)}}(f)),function(g){c(g)})})};
W.resolve=function(a){return new W(function(b,c){a instanceof W?a.then(b,c):b(a)})};
W.reject=function(a){return new W(function(b,c){c(a)})};
W.prototype.then=function(a,b){var c=this,d=null!==a&&void 0!==a?a:xg,e=null!==b&&void 0!==b?b:wg;return new W(function(f,g){"PENDING"===c.state.status?(c.h.push(function(){zg(c,c,d,f,g)}),c.onRejected.push(function(){Ag(c,c,e,f,g)})):"FULFILLED"===c.state.status?zg(c,c,d,f,g):"REJECTED"===c.state.status&&Ag(c,c,e,f,g)})};
function yg(a,b){a.then(void 0,b)}
function zg(a,b,c,d,e){try{if("FULFILLED"!==a.state.status)throw Error("calling handleResolve before the promise is fulfilled.");var f=c(a.state.value);f instanceof W?Bg(a,b,f,d,e):d(f)}catch(g){e(g)}}
function Ag(a,b,c,d,e){try{if("REJECTED"!==a.state.status)throw Error("calling handleReject before the promise is rejected.");var f=c(a.state.reason);f instanceof W?Bg(a,b,f,d,e):d(f)}catch(g){e(g)}}
function Bg(a,b,c,d,e){b===c?e(new TypeError("Circular promise chain detected.")):c.then(function(f){f instanceof W?Bg(a,b,f,d,e):d(f)},function(f){e(f)})}
;function Cg(a,b,c){function d(){c(a.error);f()}
function e(){b(a.result);f()}
function f(){try{a.removeEventListener("success",e),a.removeEventListener("error",d)}catch(g){}}
a.addEventListener("success",e);a.addEventListener("error",d)}
function Dg(a){return new Promise(function(b,c){Cg(a,b,c)})}
function X(a){return new W(function(b,c){Cg(a,b,c)})}
;function Eg(a,b){return new W(function(c,d){function e(){var f=a?b(a):null;f?f.then(function(g){a=g;e()},d):c()}
e()})}
;function Fg(a,b){this.h=a;this.options=b;this.transactionCount=0;this.j=Math.round(U());this.i=!1}
n=Fg.prototype;n.add=function(a,b,c){return Gg(this,[a],{mode:"readwrite",D:!0},function(d){return Hg(d,a).add(b,c)})};
n.clear=function(a){return Gg(this,[a],{mode:"readwrite",D:!0},function(b){return Hg(b,a).clear()})};
n.close=function(){var a;this.h.close();(null===(a=this.options)||void 0===a?0:a.closed)&&this.options.closed()};
n.count=function(a,b){return Gg(this,[a],{mode:"readonly",D:!0},function(c){return Hg(c,a).count(b)})};
function Ig(a,b,c){a=a.h.createObjectStore(b,c);return new Jg(a)}
n.delete=function(a,b){return Gg(this,[a],{mode:"readwrite",D:!0},function(c){return Hg(c,a).delete(b)})};
n.get=function(a,b){return Gg(this,[a],{mode:"readonly",D:!0},function(c){return Hg(c,a).get(b)})};
function Kg(a,b){return Gg(a,["LogsRequestsStore"],{mode:"readwrite",D:!0},function(c){c=Hg(c,"LogsRequestsStore");return X(c.h.put(b,void 0))})}
n.objectStoreNames=function(){return Array.from(this.h.objectStoreNames)};
function Gg(a,b,c,d){return J(a,function f(){var g=this,h,k,l,m,p,r,q,y,C,A,Q,R;return x(f,function(E){switch(E.h){case 1:var N={mode:"readonly",D:!1};"string"===typeof c?N.mode=c:N=c;h=N;g.transactionCount++;k=h.D?Ie("ytidb_transaction_try_count",1):1;l=0;case 2:if(m){E.u(3);break}l++;p=Math.round(U());pa(E,4);r=g.h.transaction(b,h.mode);N=new Lg(r);N=Mg(N,d);return w(E,N,6);case 6:return q=E.i,y=Math.round(U()),Ng(g,p,y,l,void 0,b.join(),h),E.return(q);case 4:C=qa(E);A=Math.round(U());Q=vg(C,g.h.name,
b.join());if((R=Q instanceof V&&!Q.h)||l>=k)Ng(g,p,A,l,Q,b.join(),h),m=Q;E.u(2);break;case 3:return E.return(Promise.reject(m))}})})}
function Ng(a,b,c,d,e,f,g){b=c-b;e?(e instanceof V&&("QUOTA_EXCEEDED"===e.type||"QUOTA_MAYBE_EXCEEDED"===e.type)&&ig("QUOTA_EXCEEDED",{dbName:mg(a.h.name),objectStoreNames:f,transactionCount:a.transactionCount,transactionMode:g.mode}),e instanceof V&&"UNKNOWN_ABORT"===e.type&&(ig("TRANSACTION_UNEXPECTEDLY_ABORTED",{objectStoreNames:f,transactionDuration:b,transactionCount:a.transactionCount,dbDuration:c-a.j}),a.i=!0),Og(a,!1,d,f,b),hg(e)):Og(a,!0,d,f,b)}
function Og(a,b,c,d,e){ig("TRANSACTION_ENDED",{objectStoreNames:d,connectionHasUnknownAbortedTransaction:a.i,duration:e,isSuccessful:b,tryCount:c})}
n.getName=function(){return this.h.name};
function Jg(a){this.h=a}
n=Jg.prototype;n.add=function(a,b){return X(this.h.add(a,b))};
n.autoIncrement=function(){return this.h.autoIncrement};
n.clear=function(){return X(this.h.clear()).then(function(){})};
n.count=function(a){return X(this.h.count(a))};
function Pg(a,b){return Qg(a,{query:b},function(c){return c.delete().then(function(){return c.continue()})}).then(function(){})}
n.delete=function(a){return a instanceof IDBKeyRange?Pg(this,a):X(this.h.delete(a))};
n.get=function(a){return X(this.h.get(a))};
n.index=function(a){return new Rg(this.h.index(a))};
n.getName=function(){return this.h.name};
n.keyPath=function(){return this.h.keyPath};
function Qg(a,b,c){a=a.h.openCursor(b.query,b.direction);return Sg(a).then(function(d){return Eg(d,c)})}
function Lg(a){var b=this;this.h=a;this.i=new Map;this.aborted=!1;this.done=new Promise(function(c,d){b.h.addEventListener("complete",function(){c()});
b.h.addEventListener("error",function(e){e.currentTarget===e.target&&d(b.h.error)});
b.h.addEventListener("abort",function(){var e=b.h.error;if(e)d(e);else if(!b.aborted){e=V;for(var f=b.h.objectStoreNames,g=[],h=0;h<f.length;h++){var k=f.item(h);if(null===k)throw Error("Invariant: item in DOMStringList is null");g.push(k)}e=new e("UNKNOWN_ABORT",{objectStoreNames:g.join(),dbName:b.h.db.name,mode:b.h.mode});d(e)}})})}
function Mg(a,b){var c=new Promise(function(d,e){try{yg(b(a).then(function(f){d(f)}),e)}catch(f){e(f),a.abort()}});
return Promise.all([c,a.done]).then(function(d){return u(d).next().value})}
Lg.prototype.abort=function(){this.h.abort();this.aborted=!0;throw new V("EXPLICIT_ABORT");};
function Hg(a,b){b=a.h.objectStore(b);var c=a.i.get(b);c||(c=new Jg(b),a.i.set(b,c));return c}
function Rg(a){this.h=a}
n=Rg.prototype;n.count=function(a){return X(this.h.count(a))};
n.delete=function(a){return Tg(this,{query:a},function(b){return b.delete().then(function(){return b.continue()})})};
n.get=function(a){return X(this.h.get(a))};
n.getKey=function(a){return X(this.h.getKey(a))};
n.keyPath=function(){return this.h.keyPath};
n.unique=function(){return this.h.unique};
function Tg(a,b,c){a=a.h.openCursor(void 0===b.query?null:b.query,void 0===b.direction?"next":b.direction);return Sg(a).then(function(d){return Eg(d,c)})}
function Ug(a,b){this.request=a;this.cursor=b}
function Sg(a){return X(a).then(function(b){return null===b?null:new Ug(a,b)})}
n=Ug.prototype;n.advance=function(a){this.cursor.advance(a);return Sg(this.request)};
n.continue=function(a){this.cursor.continue(a);return Sg(this.request)};
n.delete=function(){return X(this.cursor.delete()).then(function(){})};
n.getKey=function(){return this.cursor.key};
n.update=function(a){return X(this.cursor.update(a))};function Vg(a,b,c){return new Promise(function(d,e){function f(){r||(r=new Fg(g.result,{closed:p}));return r}
var g=self.indexedDB.open(a,b),h=c.blocked,k=c.blocking,l=c.Ja,m=c.upgrade,p=c.closed,r;g.addEventListener("upgradeneeded",function(q){try{if(null===q.newVersion)throw Error("Invariant: newVersion on IDbVersionChangeEvent is null");if(null===g.transaction)throw Error("Invariant: transaction on IDbOpenDbRequest is null");q.dataLoss&&"none"!==q.dataLoss&&ig("IDB_DATA_CORRUPTED",{reason:q.dataLossMessage||"unknown reason",dbName:mg(a)});var y=f(),C=new Lg(g.transaction);m&&m(y,q.oldVersion,q.newVersion,
C);C.done.catch(function(A){e(A)})}catch(A){e(A)}});
g.addEventListener("success",function(){var q=g.result;k&&q.addEventListener("versionchange",function(){k(f())});
q.addEventListener("close",function(){ig("IDB_UNEXPECTEDLY_CLOSED",{dbName:mg(a),dbVersion:q.version});l&&l()});
d(f())});
g.addEventListener("error",function(){e(g.error)});
h&&g.addEventListener("blocked",function(){h()})})}
function Wg(a,b,c){c=void 0===c?{}:c;return Vg(a,b,c)}
function Xg(a,b){b=void 0===b?{}:b;return J(this,function d(){var e,f,g;return x(d,function(h){e=self.indexedDB.deleteDatabase(a);f=b;(g=f.blocked)&&e.addEventListener("blocked",function(){g()});
return w(h,Dg(e),0)})})}
;function Yg(a,b){this.name=a;this.options=b;this.j=!1}
Yg.prototype.i=function(a,b,c){c=void 0===c?{}:c;return Wg(a,b,c)};
Yg.prototype.delete=function(a){a=void 0===a?{}:a;return Xg(this.name,a)};
Yg.prototype.open=function(){var a=this;if(!this.h){var b,c=function(){a.h===b&&(a.h=void 0)},d={blocking:function(f){f.close()},
closed:c,Ja:c,upgrade:this.options.upgrade},e=function(){return J(a,function g(){var h=this,k,l,m;return x(g,function(p){switch(p.h){case 1:return pa(p,2),w(p,h.i(h.name,h.options.version,d),4);case 4:k=p.i;a:{var r=u(Object.keys(h.options.oa));for(var q=r.next();!q.done;q=r.next())if(q=q.value,!k.h.objectStoreNames.contains(q)){r=q;break a}r=void 0}l=r;if(void 0===l){p.u(5);break}if(h.j){p.u(6);break}h.j=!0;return w(p,h.delete(),7);case 7:return p.return(e());case 6:throw new tg(l);case 5:return p.return(k);
case 2:m=qa(p);if(m instanceof DOMException?"VersionError"===m.name:"DOMError"in self&&m instanceof DOMError?"VersionError"===m.name:m instanceof Object&&"message"in m&&"An attempt was made to open a database using a lower version than the existing version."===m.message)return p.return(h.i(h.name,void 0,Object.assign(Object.assign({},d),{upgrade:void 0})));c();throw m;}})})};
this.h=b=e()}return this.h};var Zg=new Yg("YtIdbMeta",{oa:{databases:!0},upgrade:function(a,b){1>b&&Ig(a,"databases",{keyPath:"actualName"})}});
function $g(a){return J(this,function c(){var d;return x(c,function(e){if(1==e.h)return w(e,Zg.open(),2);d=e.i;return e.return(Gg(d,["databases"],{D:!0,mode:"readwrite"},function(f){var g=Hg(f,"databases");return g.get(a.actualName).then(function(h){if(h?a.actualName!==h.actualName||a.publicName!==h.publicName||a.userIdentifier!==h.userIdentifier||a.clearDataOnAuthChange!==h.clearDataOnAuthChange:1)return X(g.h.put(a,void 0)).then(function(){})})}))})})}
function ah(a){return J(this,function c(){var d;return x(c,function(e){if(1==e.h)return w(e,Zg.open(),2);d=e.i;return e.return(d.delete("databases",a))})})}
;var bh;
function ch(){return J(this,function b(){var c,d,e;return x(b,function(f){switch(f.h){case 1:if(T("ytidb_is_supported_cache_success_result")&&(c=dg(),null===c||void 0===c?0:c.hasSucceededOnce))return f.return(new eg(!0));var g;if(g=ff)g=/WebKit\/([0-9]+)/.exec(lb),g=!!(g&&600<=parseInt(g[1],10));g&&(g=/WebKit\/([0-9]+)/.exec(lb),g=!(g&&602<=parseInt(g[1],10)));if(g||Cb)return f.return(new eg(!1));try{if(d=self,!(d.indexedDB&&d.IDBIndex&&d.IDBKeyRange&&d.IDBObjectStore))return f.return(new eg(!1))}catch(h){return f.return(new eg(!1))}if(!("IDBTransaction"in self&&
"objectStoreNames"in IDBTransaction.prototype))return f.return(new eg(!1));pa(f,2);e={actualName:"yt-idb-test-do-not-use",publicName:"yt-idb-test-do-not-use",userIdentifier:void 0};return w(f,$g(e),4);case 4:return w(f,ah("yt-idb-test-do-not-use"),5);case 5:return f.return(new eg(!0));case 2:return qa(f),f.return(new eg(!1))}})})}
function dh(){if(void 0!==bh)return bh;gg=!0;return bh=ch().then(function(a){gg=!1;return a.isSupported()})}
;function eh(a){try{kg();var b=!0}catch(c){b=!1}if(!b)throw a=new V("AUTH_INVALID"),hg(a),a;b=kg();return{actualName:a+":"+b,publicName:a,userIdentifier:b}}
function fh(a,b,c,d){var e,f;return J(this,function h(){var k,l;return x(h,function(m){switch(m.h){case 1:return w(m,gh({caller:"openDbImpl",publicName:a,version:b}),2);case 2:return lg(a),k=c?{actualName:a,publicName:a,userIdentifier:void 0}:eh(a),k.clearDataOnAuthChange=T("remove_clear_data_on_auth_change_killswitch")?null!==(e=d.clearDataOnAuthChange)&&void 0!==e?e:!1:null!==(f=d.clearDataOnAuthChange)&&void 0!==f?f:!0,pa(m,3),w(m,$g(k),5);case 5:return w(m,Wg(k.actualName,b,d),6);case 6:return m.return(m.i);
case 3:return l=qa(m),pa(m,7),w(m,ah(k.actualName),9);case 9:m.h=8;m.s=0;break;case 7:qa(m);case 8:throw l;}})})}
function gh(a){return J(this,function c(){var d;return x(c,function(e){if(1==e.h)return w(e,dh(),2);if(!e.i)throw d=new V("IDB_NOT_SUPPORTED",{context:a}),hg(d),d;e.h=0})})}
function hh(a,b,c){c=void 0===c?{}:c;return fh(a,b,!1,c)}
function ih(a,b,c){c=void 0===c?{}:c;return fh(a,b,!0,c)}
function jh(a,b){b=void 0===b?{}:b;return J(this,function d(){var e;return x(d,function(f){if(1==f.h)return w(f,dh(),2);if(3!=f.h){if(!f.i)return f.return();lg(a);e=eh(a);return w(f,Xg(e.actualName,b),3)}return w(f,ah(e.actualName),0)})})}
function kh(a,b){b=void 0===b?{}:b;return J(this,function d(){return x(d,function(e){if(1==e.h)return w(e,dh(),2);if(3!=e.h){if(!e.i)return e.return();lg(a);return w(e,Xg(a,b),3)}return w(e,ah(a),0)})})}
;function lh(){W.call(this,function(){});
throw Error("Not allowed to instantiate the thennable outside of the core library.");}
v(lh,W);lh.reject=W.reject;lh.resolve=W.resolve;lh.all=W.all;function mh(a,b){Yg.call(this,a,b);this.options=b;lg(a)}
v(mh,Yg);function nh(a){var b;return function(){b||(b=new mh("LogsDatabaseV2",a));return b}}
mh.prototype.i=function(a,b,c){c=void 0===c?{}:c;return(this.options.ra?ih:hh)(a,b,Object.assign(Object.assign({},c),{clearDataOnAuthChange:this.options.clearDataOnAuthChange}))};
mh.prototype.delete=function(a){a=void 0===a?{}:a;return(this.options.ra?kh:jh)(this.name,a)};var oh;
function ph(){if(oh)return oh();var a={};oh=nh({oa:(a.LogsRequestsStore=!0,a.sapisid=!0,a.SWHealthLog=!0,a),ra:!T("nwl_use_ytidb_partitioning"),clearDataOnAuthChange:T("nwl_use_ytidb_partitioning"),upgrade:function(b,c,d){2>c&&2<=d&&(Ig(b,"LogsRequestsStore",{keyPath:"id",autoIncrement:!0}).h.createIndex("newRequest",["status","authHash","interface","timestamp"],{unique:!1}),Ig(b,"sapisid"));3>c&&3<=d&&Ig(b,"SWHealthLog",{keyPath:"id",autoIncrement:!0}).h.createIndex("swHealthNewRequest",["interface","timestamp"],
{unique:!1})},
version:3});return oh()}
;function qh(a){return J(this,function c(){var d,e,f,g,h;return x(c,function(k){switch(k.h){case 1:return d={startTime:U(),transactionType:"YT_IDB_TRANSACTION_TYPE_WRITE"},w(k,rh(),2);case 2:return e=k.i,w(k,ph().open(),3);case 3:return f=k.i,g=Object.assign(Object.assign({},a),{options:JSON.parse(JSON.stringify(a.options)),authHash:e,interface:S("INNERTUBE_CONTEXT_CLIENT_NAME",0)}),w(k,Kg(f,g),4);case 4:return h=k.i,d.Ka=U(),sh(d),k.return(h)}})})}
function th(){return J(this,function b(){var c,d,e,f,g,h,k,l;return x(b,function(m){switch(m.h){case 1:return c={startTime:U(),transactionType:"YT_IDB_TRANSACTION_TYPE_READ"},w(m,rh(),2);case 2:return d=m.i,e=S("INNERTUBE_CONTEXT_CLIENT_NAME",0),f=["NEW",d,e,0],g=["NEW",d,e,U()],h=IDBKeyRange.bound(f,g),w(m,ph().open(),3);case 3:return k=m.i,l=void 0,w(m,Gg(k,["LogsRequestsStore"],{mode:"readwrite",D:!0},function(p){return Tg(Hg(p,"LogsRequestsStore").index("newRequest"),{query:h,direction:"prev"},
function(r){r.cursor.value&&(l=r.cursor.value,l.status="QUEUED",r.update(l))})}),4);
case 4:return c.Ka=U(),sh(c),m.return(l)}})})}
function uh(a){return J(this,function c(){var d;return x(c,function(e){if(1==e.h)return w(e,ph().open(),2);d=e.i;return e.return(Gg(d,["LogsRequestsStore"],{mode:"readwrite",D:!0},function(f){var g=Hg(f,"LogsRequestsStore");return g.get(a).then(function(h){if(h)return h.status="QUEUED",X(g.h.put(h,void 0)).then(function(){return h})})}))})})}
function vh(a){return J(this,function c(){var d;return x(c,function(e){if(1==e.h)return w(e,ph().open(),2);d=e.i;return e.return(Gg(d,["LogsRequestsStore"],{mode:"readwrite",D:!0},function(f){var g=Hg(f,"LogsRequestsStore");return g.get(a).then(function(h){return h?(h.status="NEW",h.sendCount+=1,X(g.h.put(h,void 0)).then(function(){return h})):lh.resolve(void 0)})}))})})}
function wh(a){return J(this,function c(){var d;return x(c,function(e){if(1==e.h)return w(e,ph().open(),2);d=e.i;return e.return(d.delete("LogsRequestsStore",a))})})}
function rh(){return J(this,function b(){var c;return x(b,function(d){if(1==d.h){bg.h||(bg.h=new bg);var e={};var f=lc([]);f&&(e.Authorization=f,f=void 0,void 0===f&&(f=Number(S("SESSION_INDEX",0)),f=isNaN(f)?0:f),e["X-Goog-AuthUser"]=f,"INNERTUBE_HOST_OVERRIDE"in he||(e["X-Origin"]=window.location.origin),T("pageid_as_header_web")&&"DELEGATED_SESSION_ID"in he&&(e["X-Goog-PageId"]=S("DELEGATED_SESSION_ID")));e instanceof O||(f=new O(Da),yd(f,2,e),e=f);return w(d,e,2)}c=d.i;e=d.return;f=Uf(c);var g=
new Fc;g.update(JSON.stringify(f,Object.keys(f).sort()));f=g.digest();g="";for(var h=0;h<f.length;h++)g+="0123456789ABCDEF".charAt(Math.floor(f[h]/16))+"0123456789ABCDEF".charAt(f[h]%16);return e.call(d,g)})})}
function sh(a){var b=Ie("nwl_latency_sampling_rate",.01);!(.02<b)&&Math.random()<=b&&(b=B("ytPubsub2Pubsub2Instance"))&&b.publish.call(b,"nwl_transaction_latency_payload".toString(),"nwl_transaction_latency_payload",a)}
;var xh;function yh(){xh||(xh=new Vf("yt.offline"));return xh}
function zh(a){if(T("offline_error_handling")){var b=yh().get("errors",!0)||{};b[a.message]={name:a.name,stack:a.stack};a.level&&(b[a.message].level=a.level);yh().set("errors",b,2592E3,!0)}}
function Ah(){if(T("offline_error_handling")){var a=yh().get("errors",!0);if(a){for(var b in a)if(a[b]){var c=new jg(b,"sent via offline_errors");c.name=a[b].name;c.stack=a[b].stack;c.level=a[b].level;me(c)}yh().set("errors",{},2592E3,!0)}}}
;var Bh=Ie("network_polling_interval",3E4);function Y(){L.call(this);this.V=0;this.o=this.j=!1;this.v=0;this.l=this.P=!1;this.h=this.Z();this.l=T("validate_network_status");Ch(this);Dh(this)}
v(Y,L);function Eh(){if(!Y.h){var a=B("yt.networkStatusManager.instance")||new Y;F("yt.networkStatusManager.instance",a);Y.h=a}return Y.h}
n=Y.prototype;n.J=function(){this.l||this.h===this.Z()||ne(new jg("NetworkStatusManager isOnline does not match window status"));return this.h};
n.Ia=function(a){this.j=!0;if(void 0===a?0:a)this.V||Fh(this)};
n.Z=function(){var a=window.navigator.onLine;return void 0===a?!0:a};
n.ya=function(){this.P=!0};
n.ba=function(a,b){return L.prototype.ba.call(this,a,b)};
function Dh(a){window.addEventListener("online",function(){return J(a,function c(){var d=this;return x(c,function(e){if(1==e.h)return d.l?w(e,d.L(),2):(d.h=!0,d.j&&M(d,"ytnetworkstatus-online"),e.u(2));Gh(d);d.P&&Ah();e.h=0})})})}
function Ch(a){window.addEventListener("offline",function(){return J(a,function c(){var d=this;return x(c,function(e){if(1==e.h)return d.l?w(e,d.L(),2):(d.h=!1,d.j&&M(d,"ytnetworkstatus-offline"),e.u(2));Gh(d);e.h=0})})})}
function Fh(a){a.V=Ke(function(){return J(a,function c(){var d=this;return x(c,function(e){if(1==e.h){if(T("trigger_nsm_validation_checks_with_nwl")&&!d.h)return w(e,d.L(),3);if(d.Z()){if(!1!==d.h)return e.u(3);d.o=!0;d.v=U();return d.j?d.l?w(e,d.L(),11):(d.h=!0,M(d,"ytnetworkstatus-online"),e.u(11)):e.u(11)}if(!0!==d.h)return e.u(3);d.o=!0;d.v=U();return d.j?d.l?w(e,d.L(),3):(d.h=!1,M(d,"ytnetworkstatus-offline"),e.u(3)):e.u(3)}if(3!=e.h)return d.P&&Ah(),e.u(3);Fh(d);e.h=0})})},Bh)}
function Gh(a){a.o&&(ne(new jg("NetworkStatusManager state did not match poll",U()-a.v)),a.o=!1)}
n.L=function(a){var b=this;return this.B?this.B:this.B=new Promise(function(c){return J(b,function e(){var f,g,h,k=this;return x(e,function(l){switch(l.h){case 1:return f=window.AbortController?new window.AbortController:void 0,g=null===f||void 0===f?void 0:f.signal,h=!1,pa(l,2,3),f&&(k.U=Me(function(){f.abort()},a||2E4)),w(l,fetch("/generate_204",{method:"HEAD",
signal:g}),5);case 5:h=!0;case 3:ra(l);k.B=void 0;k.U&&Oe(k.U);h!==k.h&&(k.h=h,k.h&&k.j?M(k,"ytnetworkstatus-online"):k.j&&M(k,"ytnetworkstatus-offline"));c(h);sa(l);break;case 2:qa(l),h=!1,l.u(3)}})})})};
Y.prototype.sendNetworkCheckRequest=Y.prototype.L;Y.prototype.listen=Y.prototype.ba;Y.prototype.enableErrorFlushing=Y.prototype.ya;Y.prototype.getWindowStatus=Y.prototype.Z;Y.prototype.monitorNetworkStatusChange=Y.prototype.Ia;Y.prototype.isNetworkAvailable=Y.prototype.J;Y.getInstance=Eh;function Hh(a){a=void 0===a?{}:a;L.call(this);var b=this;this.j=this.o=0;this.h=Eh();var c=B("yt.networkStatusManager.instance.monitorNetworkStatusChange").bind(this.h);c&&c(a.za);a.Ha&&(c=B("yt.networkStatusManager.instance.enableErrorFlushing").bind(this.h))&&c();if(c=B("yt.networkStatusManager.instance.listen").bind(this.h))a.ca?(this.ca=a.ca,c("ytnetworkstatus-online",function(){Ih(b,"publicytnetworkstatus-online")}),c("ytnetworkstatus-offline",function(){Ih(b,"publicytnetworkstatus-offline")})):
(c("ytnetworkstatus-online",function(){M(b,"publicytnetworkstatus-online")}),c("ytnetworkstatus-offline",function(){M(b,"publicytnetworkstatus-offline")}))}
v(Hh,L);Hh.prototype.J=function(){var a=B("yt.networkStatusManager.instance.isNetworkAvailable").bind(this.h);return a?a():!0};
Hh.prototype.L=function(a){return J(this,function c(){var d=this,e;return x(c,function(f){return(e=B("yt.networkStatusManager.instance.sendNetworkCheckRequest").bind(d.h))?f.return(e(a)):f.return(!0)})})};
function Ih(a,b){a.ca?a.j?(Oe(a.o),a.o=Me(function(){a.l!==b&&(M(a,b),a.l=b,a.j=U())},a.ca-(U()-a.j))):(M(a,b),a.l=b,a.j=U()):M(a,b)}
;var Jh=0,Kh=0,Lh,Mh=z.ytNetworklessLoggingInitializationOptions||{isNwlInitialized:!1,isIdbSupported:!1,potentialEsfErrorCounter:Kh};T("export_networkless_options")&&F("ytNetworklessLoggingInitializationOptions",Mh);function Nh(a,b){function c(d){var e=Oh().J();if(!Ph()||!d||e&&T("vss_networkless_bypass_write"))Qh(a,b);else{var f={url:a,options:b,timestamp:U(),status:"NEW",sendCount:0};qh(f).then(function(g){f.id=g;(Oh().J()||T("networkless_always_online"))&&Rh(f)}).catch(function(g){Rh(f);
Oh().J()?me(g):zh(g)})}}
b=void 0===b?{}:b;T("skip_is_supported_killswitch")?dh().then(function(d){c(d)}):c(Sh())}
function Th(a,b){function c(d){if(Ph()&&d){var e={url:a,options:b,timestamp:U(),status:"NEW",sendCount:0},f=!1,g=b.onSuccess?b.onSuccess:function(){};
e.options.onSuccess=function(h,k){void 0!==e.id?wh(e.id):f=!0;g(h,k)};
Qh(e.url,e.options);qh(e).then(function(h){e.id=h;f&&wh(e.id)}).catch(function(h){Oh().J()?me(h):zh(h)})}else Qh(a,b)}
b=void 0===b?{}:b;T("skip_is_supported_killswitch")?dh().then(function(d){c(d)}):c(Sh())}
function Uh(){var a=this;Jh||(Jh=Me(function(){return J(a,function c(){var d;return x(c,function(e){if(1==e.h)return w(e,th(),2);if(3!=e.h)return d=e.i,d?w(e,Rh(d),3):(Oe(Jh),Jh=0,e.return());if(!T("nwl_throttling_race_fix")||Jh)Jh=0,Uh();e.h=0})})},100))}
function Rh(a){return J(this,function c(){var d;return x(c,function(e){switch(e.h){case 1:if(void 0===a.id){e.u(2);break}return w(e,uh(a.id),3);case 3:(d=e.i)?a=d:ne(Error("The request cannot be found in the database."));case 2:var f=a.timestamp;if(!(2592E6<=U()-f)){e.u(4);break}ne(Error("Networkless Logging: Stored logs request expired age limit"));if(void 0===a.id){e.u(5);break}return w(e,wh(a.id),5);case 5:return e.return();case 4:f=a=Vh(a);var g,h;if(null===(h=null===(g=null===f||void 0===f?void 0:
f.options)||void 0===g?void 0:g.postParams)||void 0===h?0:h.requestTimeMs)f.options.postParams.requestTimeMs=Math.round(U());(a=f)&&Qh(a.url,a.options);e.h=0}})})}
function Vh(a){var b=this,c=a.options.onError?a.options.onError:function(){};
a.options.onError=function(e,f){return J(b,function h(){return x(h,function(k){switch(k.h){case 1:if(!(T("trigger_nsm_validation_checks_with_nwl")&&(B("ytNetworklessLoggingInitializationOptions")?Mh.potentialEsfErrorCounter:Kh)<=Ie("potential_esf_error_limit",10))){k.u(2);break}return w(k,Oh().L(),3);case 3:if(Oh().J())B("ytNetworklessLoggingInitializationOptions")&&Mh.potentialEsfErrorCounter++,Kh++;else return c(e,f),k.return();case 2:if(void 0===(null===a||void 0===a?void 0:a.id)){k.u(4);break}return 1>
a.sendCount?w(k,vh(a.id),8):w(k,wh(a.id),4);case 8:Me(function(){Oh().J()&&Uh()},5E3);
case 4:c(e,f),k.h=0}})})};
var d=a.options.onSuccess?a.options.onSuccess:function(){};
a.options.onSuccess=function(e,f){return J(b,function h(){return x(h,function(k){if(1==k.h)return void 0===(null===a||void 0===a?void 0:a.id)?k.u(2):w(k,wh(a.id),2);d(e,f);k.h=0})})};
return a}
function Oh(){Lh||(Lh=new Hh({Ha:!0,za:T("trigger_nsm_validation_checks_with_nwl")}));return Lh}
function Qh(a,b){if(T("networkless_with_beacon")){var c=["method","postBody"];if(Object.keys(b).length>c.length)var d=!0;else{d=0;c=u(c);for(var e=c.next();!e.done;e=c.next())b.hasOwnProperty(e.value)&&d++;d=Object.keys(b).length!==d}d?Ye(a,b):jf(a,void 0,b.postBody)}else Ye(a,b)}
function Ph(){return B("ytNetworklessLoggingInitializationOptions")?Mh.isNwlInitialized:!1}
function Sh(){return B("ytNetworklessLoggingInitializationOptions")?Mh.isIdbSupported:!1}
;function Wh(a){var b=this;this.config_=null;a?this.config_=a:Sf()&&(this.config_=Jf());Ke(function(){$f(b)},5E3)}
Wh.prototype.isReady=function(){!this.config_&&Sf()&&(this.config_=Jf());return!!this.config_};
function Mf(a,b,c,d){function e(r){r=void 0===r?!1:r;var q;if(d.retry&&"www.youtube-nocookie.com"!=h&&(r||(q=Yf(b,c,l,k)),q)){var y=g.onSuccess,C=g.onFetchSuccess;g.onSuccess=function(A,Q){Zf(q);y(A,Q)};
c.onFetchSuccess=function(A,Q){Zf(q);C(A,Q)}}try{r&&d.retry&&!d.na.bypassNetworkless?(g.method="POST",!d.na.writeThenSend&&T("nwl_send_fast_on_unload")?Th(p,g):Nh(p,g)):(g.method="POST",g.postParams||(g.postParams={}),Ye(p,g))}catch(A){if("InvalidAccessError"==A.name)q&&(Zf(q),q=0),ne(Error("An extension is blocking network request."));
else throw A;}q&&Ke(function(){$f(a)},5E3)}
!S("VISITOR_DATA")&&"visitor_id"!==b&&.01>Math.random()&&ne(new jg("Missing VISITOR_DATA when sending innertube request.",b,c,d));if(!a.isReady()){var f=new jg("innertube xhrclient not ready",b,c,d);me(f);throw f;}var g={headers:{"Content-Type":"application/json"},method:"POST",postParams:c,postBodyFormat:"JSON",onTimeout:function(){d.onTimeout()},
onFetchTimeout:d.onTimeout,onSuccess:function(r,q){if(d.onSuccess)d.onSuccess(q)},
onFetchSuccess:function(r){if(d.onSuccess)d.onSuccess(r)},
onError:function(r,q){if(d.onError)d.onError(q)},
onFetchError:function(r){if(d.onError)d.onError(r)},
timeout:d.timeout,withCredentials:!0},h="";(f=a.config_.Ea)&&(h=f);var k=a.config_.Ga||!1,l=Tf(k,h,d);Object.assign(g.headers,l);g.headers.Authorization&&!h&&(g.headers["x-origin"]=window.location.origin);f="/youtubei/"+a.config_.innertubeApiVersion+"/"+b;var m={alt:"json"};a.config_.Fa&&g.headers.Authorization||(m.key=a.config_.innertubeApiKey);var p=Fe(""+h+f,m||{},!0);Ph()?dh().then(function(r){e(r)}):e(!1)}
;function Xh(a,b){var c=void 0===c?{}:c;var d=Wh;S("ytLoggingEventsDefaultDisabled",!1)&&Wh==Wh&&(d=null);c=void 0===c?{}:c;var e={},f=Math.round(c.timestamp||U());e.eventTimeMs=f<Number.MAX_SAFE_INTEGER?f:0;e[a]=b;a=B("_lact",window);a=null==a?-1:Math.max(Date.now()-a,0);e.context={lastActivityMs:String(c.timestamp||!isFinite(a)?-1:a)};T("log_sequence_info_on_gel_web")&&c.qa&&(a=e.context,b=c.qa,Nf[b]=b in Nf?Nf[b]+1:0,a.sequence={index:Nf[b],groupKey:b},c.Za&&delete Nf[c.qa]);(c.fb?Gf:Cf)({endpoint:"log_event",
payload:e,H:c.H,Y:c.Y},d)}
;var Yh=[{la:function(a){return"Cannot read property '"+a.key+"'"},
fa:{TypeError:[{regexp:/Cannot read property '([^']+)' of (null|undefined)/,groups:["key","value"]},{regexp:/\u65e0\u6cd5\u83b7\u53d6\u672a\u5b9a\u4e49\u6216 (null|undefined) \u5f15\u7528\u7684\u5c5e\u6027\u201c([^\u201d]+)\u201d/,groups:["value","key"]},{regexp:/\uc815\uc758\ub418\uc9c0 \uc54a\uc74c \ub610\ub294 (null|undefined) \ucc38\uc870\uc778 '([^']+)' \uc18d\uc131\uc744 \uac00\uc838\uc62c \uc218 \uc5c6\uc2b5\ub2c8\ub2e4./,groups:["value","key"]},{regexp:/No se puede obtener la propiedad '([^']+)' de referencia nula o sin definir/,
groups:["key"]},{regexp:/Unable to get property '([^']+)' of (undefined or null) reference/,groups:["key","value"]}],Error:[{regexp:/(Permission denied) to access property "([^']+)"/,groups:["reason","key"]}]}},{la:function(a){return"Cannot call '"+a.key+"'"},
fa:{TypeError:[{regexp:/(?:([^ ]+)?\.)?([^ ]+) is not a function/,groups:["base","key"]},{regexp:/([^ ]+) called on (null or undefined)/,groups:["key","value"]},{regexp:/Object (.*) has no method '([^ ]+)'/,groups:["base","key"]},{regexp:/Object doesn't support property or method '([^ ]+)'/,groups:["key"]},{regexp:/\u30aa\u30d6\u30b8\u30a7\u30af\u30c8\u306f '([^']+)' \u30d7\u30ed\u30d1\u30c6\u30a3\u307e\u305f\u306f\u30e1\u30bd\u30c3\u30c9\u3092\u30b5\u30dd\u30fc\u30c8\u3057\u3066\u3044\u307e\u305b\u3093/,
groups:["key"]},{regexp:/\uac1c\uccb4\uac00 '([^']+)' \uc18d\uc131\uc774\ub098 \uba54\uc11c\ub4dc\ub97c \uc9c0\uc6d0\ud558\uc9c0 \uc54a\uc2b5\ub2c8\ub2e4./,groups:["key"]}]}}];var $h={K:[],I:[{va:Zh,weight:500}]};function Zh(a){a=a.stack;return a.includes("chrome://")||a.includes("chrome-extension://")||a.includes("moz-extension://")}
;function ai(){this.I=[];this.K=[]}
var bi;function ci(){if(!bi){var a=bi=new ai;a.K.length=0;a.I.length=0;$h.K&&a.K.push.apply(a.K,$h.K);$h.I&&a.I.push.apply(a.I,$h.I)}return bi}
;var di=new P;function ei(a){function b(){return a.charCodeAt(d++)}
var c=a.length,d=0;do{var e=fi(b);if(Infinity===e)break;var f=e>>3;switch(e&7){case 0:e=fi(b);if(2===f)return e;break;case 1:if(2===f)return;d+=8;break;case 2:e=fi(b);if(2===f)return a.substr(d,e);d+=e;break;case 5:if(2===f)return;d+=4;break;default:return}}while(d<c)}
function fi(a){var b=a(),c=b&127;if(128>b)return c;b=a();c|=(b&127)<<7;if(128>b)return c;b=a();c|=(b&127)<<14;if(128>b)return c;b=a();return 128>b?c|(b&127)<<21:Infinity}
;function gi(a,b,c,d){if(a)if(Array.isArray(a)){var e=d;for(d=0;d<a.length&&!(a[d]&&(e+=hi(d,a[d],b,c),500<e));d++);d=e}else if("object"===typeof a)for(e in a){if(a[e]){var f=e;var g=a[e],h=b,k=c;f="string"!==typeof g||"clickTrackingParams"!==f&&"trackingParams"!==f?0:(g=ei(atob(g.replace(/-/g,"+").replace(/_/g,"/"))))?hi(f+".ve",g,h,k):0;d+=f;d+=hi(e,a[e],b,c);if(500<d)break}}else c[b]=ii(a),d+=c[b].length;else c[b]=ii(a),d+=c[b].length;return d}
function hi(a,b,c,d){c+="."+a;a=ii(b);d[c]=a;return c.length+a.length}
function ii(a){return("string"===typeof a?a:String(JSON.stringify(a))).substr(0,500)}
;var ji=new Set,ki=0,li=0,mi=0,ni=[],oi=["PhantomJS","Googlebot","TO STOP THIS SECURITY SCAN go/scan"];var pi={};function qi(a){return pi[a]||(pi[a]=String(a).replace(/\-([a-z])/g,function(b,c){return c.toUpperCase()}))}
;var ri={},si=[],Od=new P,ti={};function ui(){for(var a=u(si),b=a.next();!b.done;b=a.next())b=b.value,b()}
function vi(a,b){var c;"yt:"===a.tagName.toLowerCase().substr(0,3)?c=a.getAttribute(b):c=a?a.dataset?a.dataset[qi(b)]:a.getAttribute("data-"+b):null;return c}
function wi(a,b){for(var c=1;c<arguments.length;++c);Od.O.apply(Od,arguments)}
;function xi(a){this.j=this.h=!1;this.i=a||{};a=document.getElementById("www-widgetapi-script");if(this.h=!!("https:"===document.location.protocol||a&&0===a.src.indexOf("https:"))){a=[this.i,window.YTConfig||{}];for(var b=0;b<a.length;b++)a[b].host&&(a[b].host=a[b].host.toString().replace("http://","https://"))}}
function Z(a,b){a=[a.i,window.YTConfig||{}];for(var c=0;c<a.length;c++){var d=a[c][b];if(void 0!==d)return d}return null}
function yi(a,b,c){zi||(zi={},ve(window,"message",function(d){a:{if(d.origin===Z(a,"host")||d.origin===Z(a,"host").toString().replace(/^http:/,"https:")){try{var e=JSON.parse(d.data)}catch(f){e=void 0;break a}a.j=!0;a.h||0!==d.origin.indexOf("https:")||(a.h=!0);if(d=zi[e.id])d.o=!0,d.o&&(H(d.s,d.sendMessage,d),d.s.length=0),d.ga(e)}e=void 0}return e}));
zi[c]=b}
var zi=null;function Ai(a,b,c){this.m=this.h=this.i=null;this.j=0;this.o=!1;this.s=[];this.l=null;this.B={};if(!a)throw Error("YouTube player element ID required.");this.id=Fa(this);this.v=c;this.setup(a,b)}
n=Ai.prototype;n.setSize=function(a,b){this.h.width=a.toString();this.h.height=b.toString();return this};
n.ta=function(){return this.h};
n.ga=function(a){Bi(this,a.event,a)};
n.addEventListener=function(a,b){var c=b;"string"===typeof b&&(c=function(){window[b].apply(window,arguments)});
if(!c)return this;this.l.subscribe(a,c);Ci(this,a);return this};
function Di(a,b){b=b.split(".");if(2===b.length){var c=b[1];a.v===b[0]&&Ci(a,c)}}
n.destroy=function(){this.h&&this.h.id&&(ri[this.h.id]=null);var a=this.l;a&&"function"==typeof a.dispose&&a.dispose();if(this.m){a=this.h;var b=a.parentNode;b&&b.replaceChild(this.m,a)}else(a=this.h)&&a.parentNode&&a.parentNode.removeChild(a);zi&&(zi[this.id]=null);this.i=null;a=this.h;for(var c in Ya)Ya[c][0]==a&&te(c);this.m=this.h=null};
n.ia=function(){return{}};
function Ei(a,b,c){c=c||[];c=Array.prototype.slice.call(c);b={event:"command",func:b,args:c};a.o?a.sendMessage(b):a.s.push(b)}
function Bi(a,b,c){a.l.m||(c={target:a,data:c},a.l.O(b,c),wi(a.v+"."+b,c))}
function Fi(a,b){var c=document.createElement("iframe");b=b.attributes;for(var d=0,e=b.length;d<e;d++){var f=b[d].value;null!=f&&""!==f&&"null"!==f&&c.setAttribute(b[d].name,f)}c.setAttribute("frameBorder","0");c.setAttribute("allowfullscreen","1");c.setAttribute("allow","accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture");c.setAttribute("title","YouTube "+Z(a.i,"title"));(b=Z(a.i,"width"))&&c.setAttribute("width",b.toString());(b=Z(a.i,"height"))&&c.setAttribute("height",
b.toString());var g=a.ia();g.enablejsapi=window.postMessage?1:0;window.location.host&&(g.origin=window.location.protocol+"//"+window.location.host);g.widgetid=a.id;window.location.href&&H(["debugjs","debugcss"],function(h){var k=xb(window.location.href,h);null!==k&&(g[h]=k)});
window.yt_embedsTokenValue&&(g.embedsTokenValue=encodeURIComponent(window.yt_embedsTokenValue),delete window.yt_embedsTokenValue);c.src=Z(a.i,"host")+("/embed/"+Z(a.i,"videoId"))+"?"+vb(g);return c}
n.pa=function(){this.h&&this.h.contentWindow?this.sendMessage({event:"listening"}):window.clearInterval(this.j)};
function Gi(a){yi(a.i,a,a.id);a.j=xe(a.pa.bind(a));ve(a.h,"load",function(){window.clearInterval(a.j);a.j=xe(a.pa.bind(a))})}
n.setup=function(a,b){var c=document;if(a="string"===typeof a?c.getElementById(a):a)if(c="iframe"===a.tagName.toLowerCase(),b.host||(b.host=c?tb(a.src):"https://www.youtube.com"),this.i=new xi(b),c||(b=Fi(this,a),this.m=a,(c=a.parentNode)&&c.replaceChild(b,a),a=b),this.h=a,this.h.id||(this.h.id="widget"+Fa(this.h)),ri[this.h.id]=this,window.postMessage){this.l=new P;Gi(this);b=Z(this.i,"events");for(var d in b)b.hasOwnProperty(d)&&this.addEventListener(d,b[d]);for(var e in ti)ti.hasOwnProperty(e)&&
Di(this,e)}};
function Ci(a,b){a.B[b]||(a.B[b]=!0,Ei(a,"addEventListener",[b]))}
n.sendMessage=function(a){a.id=this.id;a.channel="widget";a=xd(a);var b=this.i;var c=tb(this.h.src||"");b=0===c.indexOf("https:")?[c]:b.h?[c.replace("http:","https:")]:b.j?[c]:[c,c.replace("http:","https:")];if(this.h.contentWindow)for(c=0;c<b.length;c++)try{this.h.contentWindow.postMessage(a,b[c])}catch(A){if(A.name&&"SyntaxError"===A.name){if(!(A.message&&0<A.message.indexOf("target origin ''"))){var d=void 0,e=A;d=void 0===d?{}:d;d.name=S("INNERTUBE_CONTEXT_CLIENT_NAME",1);d.version=S("INNERTUBE_CONTEXT_CLIENT_VERSION",
void 0);var f=d||{};d="WARNING";d=void 0===d?"ERROR":d;if(e){e.hasOwnProperty("level")&&e.level&&(d=e.level);if(T("console_log_js_exceptions")){var g=e,h=[];h.push("Name: "+g.name);h.push("Message: "+g.message);g.hasOwnProperty("params")&&h.push("Error Params: "+JSON.stringify(g.params));g.hasOwnProperty("args")&&h.push("Error args: "+JSON.stringify(g.args));h.push("File name: "+g.fileName);h.push("Stacktrace: "+g.stack);window.console.log(h.join("\n"),g)}if(!(5<=ki)){g=void 0;var k=f,l=Hc(e);f=l.message||
"Unknown Error";h=l.name||"UnknownError";var m=l.stack||e.i||"Not available";if(m.startsWith(h+": "+f)){var p=m.split("\n");p.shift();m=p.join("\n")}p=l.lineNumber||"Not available";l=l.fileName||"Not available";var r=0;if(e.hasOwnProperty("args")&&e.args&&e.args.length)for(g=0;g<e.args.length&&!(r=gi(e.args[g],"params."+g,k,r),500<=r);g++);else if(e.hasOwnProperty("params")&&e.params){var q=e.params;if("object"===typeof e.params)for(g in q){if(q[g]){var y="params."+g,C=ii(q[g]);k[y]=C;r+=y.length+
C.length;if(500<r)break}}else k.params=ii(q)}if(ni.length)for(g=0;g<ni.length&&!(r=gi(ni[g],"params.context."+g,k,r),500<=r);g++);navigator.vendor&&!k.hasOwnProperty("vendor")&&(k["device.vendor"]=navigator.vendor);g={message:f,name:h,lineNumber:p,fileName:l,stack:m,params:k,sampleWeight:1};f=Number(e.columnNumber);isNaN(f)||(g.lineNumber=g.lineNumber+":"+f);if("IGNORED"===e.level)e=0;else a:{e=ci();f=u(e.K);for(h=f.next();!h.done;h=f.next())if(h=h.value,g.message&&g.message.match(h.ab)){e=h.weight;
break a}e=u(e.I);for(f=e.next();!f.done;f=e.next())if(f=f.value,f.va(g)){e=f.weight;break a}e=1}g.sampleWeight=e;e=g;g=u(Yh);for(f=g.next();!f.done;f=g.next())if(f=f.value,f.fa[e.name])for(p=u(f.fa[e.name]),h=p.next();!h.done;h=p.next())if(l=h.value,h=e.message.match(l.regexp)){e.params["params.error.original"]=h[0];p=l.groups;l={};for(m=0;m<p.length;m++)l[p[m]]=h[m+1],e.params["params.error."+p[m]]=h[m+1];e.message=f.la(l);break}e.params||(e.params={});g=ci();e.params["params.errorServiceSignature"]=
"msg="+g.K.length+"&cb="+g.I.length;e.params["params.serviceWorker"]="false";z.document&&z.document.querySelectorAll&&(e.params["params.fscripts"]=String(document.querySelectorAll("script:not([nonce])").length));window.yterr&&"function"===typeof window.yterr&&window.yterr(e);if(0!==e.sampleWeight&&!ji.has(e.message)){"ERROR"===d?(di.O("handleError",e),T("record_app_crashed_web")&&0===mi&&1===e.sampleWeight&&(mi++,Xh("appCrashed",{appCrashType:"APP_CRASH_TYPE_BREAKPAD"})),li++):"WARNING"===d&&di.O("handleWarning",
e);if(T("kevlar_gel_error_routing")){g=d;h=e;b:{f=u(oi);for(p=f.next();!p.done;p=f.next())if((l=lb)&&0<=l.toLowerCase().indexOf(p.value.toLowerCase())){f=!0;break b}f=!1}if(f)f=void 0;else{p={stackTrace:h.stack};h.fileName&&(p.filename=h.fileName);f=h.lineNumber&&h.lineNumber.split?h.lineNumber.split(":"):[];0!==f.length&&(1!==f.length||isNaN(Number(f[0]))?2!==f.length||isNaN(Number(f[0]))||isNaN(Number(f[1]))||(p.lineNumber=Number(f[0]),p.columnNumber=Number(f[1])):p.lineNumber=Number(f[0]));f={level:"ERROR_LEVEL_UNKNOWN",
message:h.message,errorClassName:h.name,sampleWeight:h.sampleWeight};"ERROR"===g?f.level="ERROR_LEVEL_ERROR":"WARNING"===g&&(f.level="ERROR_LEVEL_WARNNING");p={isObfuscated:!0,browserStackInfo:p};l={pageUrl:window.location.href,kvPairs:[]};S("FEXP_EXPERIMENTS")&&(l.experimentIds=S("FEXP_EXPERIMENTS"));if(h=h.params)for(m=u(Object.keys(h)),k=m.next();!k.done;k=m.next())k=k.value,l.kvPairs.push({key:"client."+k,value:String(h[k])});h=S("SERVER_NAME",void 0);m=S("SERVER_VERSION",void 0);h&&m&&(l.kvPairs.push({key:"server.name",
value:h}),l.kvPairs.push({key:"server.version",value:m}));f={errorMetadata:l,stackTrace:p,logMessage:f}}f&&(Xh("clientError",f),("ERROR"===g||T("errors_flush_gel_always_killswitch"))&&Ef())}if(!T("suppress_error_204_logging")){f=e;g=f.params||{};d={urlParams:{a:"logerror",t:"jserror",type:f.name,msg:f.message.substr(0,250),line:f.lineNumber,level:d,"client.name":g.name},postParams:{url:S("PAGE_NAME",window.location.href),file:f.fileName},method:"POST"};g.version&&(d["client.version"]=g.version);if(d.postParams){f.stack&&
(d.postParams.stack=f.stack);f=u(Object.keys(g));for(h=f.next();!h.done;h=f.next())h=h.value,d.postParams["client."+h]=g[h];if(g=S("LATEST_ECATCHER_SERVICE_TRACKING_PARAMS",void 0))for(f=u(Object.keys(g)),h=f.next();!h.done;h=f.next())h=h.value,d.postParams[h]=g[h];g=S("SERVER_NAME",void 0);f=S("SERVER_VERSION",void 0);g&&f&&(d.postParams["server.name"]=g,d.postParams["server.version"]=f)}Ye(S("ECATCHER_REPORT_HOST","")+"/error_204",d)}ji.add(e.message);ki++}}}}}else throw A;}else console&&console.warn&&
console.warn("The YouTube player is not attached to the DOM. API calls should be made after the onReady event. See more: https://developers.google.com/youtube/iframe_api_reference#Events")};function Hi(a){return(0===a.search("cue")||0===a.search("load"))&&"loadModule"!==a}
function Ii(a){return 0===a.search("get")||0===a.search("is")}
;function Ji(a,b){Ai.call(this,a,Object.assign({title:"video player",videoId:"",width:640,height:360},b||{}),"player");this.F={};this.playerInfo={}}
v(Ji,Ai);n=Ji.prototype;n.ia=function(){var a=Z(this.i,"playerVars");if(a){var b={},c;for(c in a)b[c]=a[c];a=b}else a={};window!==window.top&&document.referrer&&(a.widget_referrer=document.referrer.substring(0,256));if(c=Z(this.i,"embedConfig")){if(D(c))try{c=JSON.stringify(c)}catch(d){console.error("Invalid embed config JSON",d)}a.embed_config=c}return a};
n.ga=function(a){var b=a.event;a=a.info;switch(b){case "apiInfoDelivery":if(D(a))for(var c in a)a.hasOwnProperty(c)&&(this.F[c]=a[c]);break;case "infoDelivery":Ki(this,a);break;case "initialDelivery":D(a)&&(window.clearInterval(this.j),this.playerInfo={},this.F={},Li(this,a.apiInterface),Ki(this,a));break;default:Bi(this,b,a)}};
function Ki(a,b){if(D(b))for(var c in b)b.hasOwnProperty(c)&&(a.playerInfo[c]=b[c])}
function Li(a,b){H(b,function(c){this[c]||("getCurrentTime"===c?this[c]=function(){var d=this.playerInfo.currentTime;if(1===this.playerInfo.playerState){var e=(Date.now()/1E3-this.playerInfo.currentTimeLastUpdated_)*this.playerInfo.playbackRate;0<e&&(d+=Math.min(e,1))}return d}:Hi(c)?this[c]=function(){this.playerInfo={};
this.F={};Ei(this,c,arguments);return this}:Ii(c)?this[c]=function(){var d=0;
0===c.search("get")?d=3:0===c.search("is")&&(d=2);return this.playerInfo[c.charAt(d).toLowerCase()+c.substr(d+1)]}:this[c]=function(){Ei(this,c,arguments);
return this})},a)}
n.getVideoEmbedCode=function(){var a=Z(this.i,"host")+("/embed/"+Z(this.i,"videoId")),b=Number(Z(this.i,"width")),c=Number(Z(this.i,"height"));if(isNaN(b)||isNaN(c))throw Error("Invalid width or height property");b=Math.floor(b);c=Math.floor(c);kb.test(a)&&(-1!=a.indexOf("&")&&(a=a.replace(eb,"&amp;")),-1!=a.indexOf("<")&&(a=a.replace(fb,"&lt;")),-1!=a.indexOf(">")&&(a=a.replace(gb,"&gt;")),-1!=a.indexOf('"')&&(a=a.replace(hb,"&quot;")),-1!=a.indexOf("'")&&(a=a.replace(ib,"&#39;")),-1!=a.indexOf("\x00")&&
(a=a.replace(jb,"&#0;")));return'<iframe width="'+b+'" height="'+c+'" src="'+a+'" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>'};
n.getOptions=function(a){return this.F.namespaces?a?this.F[a]?this.F[a].options||[]:[]:this.F.namespaces||[]:[]};
n.getOption=function(a,b){if(this.F.namespaces&&a&&b&&this.F[a])return this.F[a][b]};
function Mi(a){if("iframe"!==a.tagName.toLowerCase()){var b=vi(a,"videoid");b&&(b={videoId:b,width:vi(a,"width"),height:vi(a,"height")},new Ji(a,b))}}
;F("YT.PlayerState.UNSTARTED",-1);F("YT.PlayerState.ENDED",0);F("YT.PlayerState.PLAYING",1);F("YT.PlayerState.PAUSED",2);F("YT.PlayerState.BUFFERING",3);F("YT.PlayerState.CUED",5);F("YT.get",function(a){return ri[a]});
F("YT.scan",ui);F("YT.subscribe",function(a,b,c){Od.subscribe(a,b,c);ti[a]=!0;for(var d in ri)ri.hasOwnProperty(d)&&Di(ri[d],a)});
F("YT.unsubscribe",function(a,b,c){Nd(a,b,c)});
F("YT.Player",Ji);Ai.prototype.destroy=Ai.prototype.destroy;Ai.prototype.setSize=Ai.prototype.setSize;Ai.prototype.getIframe=Ai.prototype.ta;Ai.prototype.addEventListener=Ai.prototype.addEventListener;Ji.prototype.getVideoEmbedCode=Ji.prototype.getVideoEmbedCode;Ji.prototype.getOptions=Ji.prototype.getOptions;Ji.prototype.getOption=Ji.prototype.getOption;
si.push(function(a){var b=a;b||(b=document);a=Ua(b.getElementsByTagName("yt:player"));var c=b||document;if(c.querySelectorAll&&c.querySelector)b=c.querySelectorAll(".yt-player");else{var d;c=document;b=b||c;if(b.querySelectorAll&&b.querySelector)b=b.querySelectorAll(".yt-player");else if(b.getElementsByClassName){var e=b.getElementsByClassName("yt-player");b=e}else{e=b.getElementsByTagName("*");var f={};for(c=d=0;b=e[c];c++){var g=b.className,h;if(h="function"==typeof g.split)h=0<=Pa(g.split(/\s+/),
"yt-player");h&&(f[d++]=b)}f.length=d;b=f}}b=Ua(b);H(Ta(a,b),Mi)});
"undefined"!=typeof YTConfig&&YTConfig.parsetags&&"onload"!=YTConfig.parsetags||ui();var Ni=z.onYTReady;Ni&&Ni();var Oi=z.onYouTubeIframeAPIReady;Oi&&Oi();var Pi=z.onYouTubePlayerAPIReady;Pi&&Pi();}).call(this);

(function(_ds){var window=this;'use strict';var B2=function(){var a=_ds.R.call(this)||this;a.eventHandler=new _ds.Ll;return a};_ds.w(B2,_ds.R);_ds.f=B2.prototype;
_ds.f.connectedCallback=function(){var a=this;this.eventHandler.listen(document.body,"devsite-sitemask-show",function(){return a.show()});this.eventHandler.listen(document.body,"keydown",function(b){"Escape"===b.key&&a.hasAttribute("visible")&&(b.preventDefault(),b.stopPropagation(),a.hb())});this.eventHandler.listen(document.body,"devsite-sitemask-hide",function(){return a.hb()});this.eventHandler.listen(this,"click",function(){return a.hb()})};
_ds.f.attributeChangedCallback=function(a,b,c){"visible"===a&&(null==c?this.dispatchEvent(new CustomEvent("devsite-sitemask-hidden",{bubbles:!0})):this.dispatchEvent(new CustomEvent("devsite-sitemask-visible",{bubbles:!0})))};_ds.f.disconnectedCallback=function(){_ds.Rl(this.eventHandler)};_ds.f.show=function(){this.setAttribute("visible",this.getAttribute("visible")||"")};_ds.f.hb=function(){this.removeAttribute("visible")};
_ds.n.Object.defineProperties(B2,{observedAttributes:{configurable:!0,enumerable:!0,get:function(){return["visible"]}}});B2.prototype.hide=B2.prototype.hb;B2.prototype.show=B2.prototype.show;B2.prototype.disconnectedCallback=B2.prototype.disconnectedCallback;B2.prototype.attributeChangedCallback=B2.prototype.attributeChangedCallback;B2.prototype.connectedCallback=B2.prototype.connectedCallback;try{window.customElements.define("devsite-sitemask",B2)}catch(a){console.warn("Unrecognized DevSite custom element - DevsiteSitemask",a)};})(_ds_www);
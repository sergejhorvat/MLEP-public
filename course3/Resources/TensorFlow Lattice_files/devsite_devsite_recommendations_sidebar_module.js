(function(_ds){var window=this;'use strict';var j1=function(a){a=a.Gd;var b='<div class="devsite-recommendations-sidebar-heading" role="heading" aria-level="2"><a href="#recommendations-link" class="devsite-nav-title devsite-recommendations-sidebar-heading-link" data-category="Site-Wide Custom Events" data-label="devsite-recommendation side-nav title" data-action="click" data-tooltip="'+_ds.Dj("See content recommendations");b+='"><svg class="devsite-recommendations-sidebar-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" aria-hidden="true"><path d=\'M12.5,8.5L10,3L7.5,8.5L2,11l5.5,2.5L10,19l2.5-5.5L18,11L12.5,8.5z M18,13l-1.25,2.75L14,17l2.75,1.25L18,21l1.25-2.75 L22,17l-2.75-1.25L18,13z\'/></svg><span class="devsite-nav-text">Recommendations</span></a></div><ul class="devsite-nav-list">';
for(var c=a.length,d=0;d<c;d++)b+=i1(a[d]);return(0,_ds.T)(b+"</ul>")},i1=function(a){a=a&&(a.N||a);var b="",c=_ds.ek(a.La(),_ds.O(a,3),_ds.O(a,4));b+='<li role="option" class="devsite-nav-item"><a href="'+_ds.X(_ds.Zj(a.getUrl()+"?"+_ds.O(a,8)))+'" class="devsite-nav-title devsite-recommendations-sidebar-title" data-category="Site-Wide Custom Events" data-label="devsite-recommendation side-nav link" data-action="click"><span class="devsite-nav-text" tooltip="">'+_ds.V(c.filter(function(d){return 0<
_ds.xj(d).length})[0])+'</span></a><div class="significatio-card-meta">Updated <span class="significatio-date" date="'+_ds.X(_ds.Rg(_ds.N(a,_ds.zB,7),1))+'"></span></div></li>';return(0,_ds.T)(b)},k1=function(a){var b=_ds.R.call(this)||this;b.timeZone=a;b.eventHandler=new _ds.Ll(b);b.j=new _ds.Tn;b.loaded=b.j.promise;b.m=new _ds.Tn;b.C=b.m.promise;b.o=null;return b},l1=function(a){a.eventHandler.listen(a,"click",function(b){b.target.classList.contains("devsite-nav-title")&&(b=b.target,a.o&&a.o.classList.remove("devsite-nav-active"),
b.classList.add("devsite-nav-active"),a.o=b)});a.eventHandler.listen(document,"devsite-on-recommendations",function(b){b=b.ma;if(null===b||void 0===b?0:b.detail)if(b=b.detail,3===b.Bd()){a.render(b);a.j.resolve();if(b=null===b||void 0===b?void 0:b.hc()){b=_ds.r(b);for(var c=b.next();!c.done;c=b.next()){c=c.value;var d=_ds.N(c,_ds.mK,10);d&&(d={targetPage:c.getUrl(),targetRank:_ds.Rg(d,2),targetType:_ds.ug(d,3,0),targetIdenticalDescriptions:_ds.Rg(d,4),targetTitleWords:_ds.Rg(d,5),targetDescriptionWords:_ds.Rg(d,
6),experiment:_ds.O(d,7)},c={category:"Site-Wide Custom Events",action:"recommended-right-nav",label:c.getUrl(),additionalParams:{recommendations:d}},a.dispatchEvent(new CustomEvent("devsite-analytics-observation",{detail:c,bubbles:!0})))}a.m.resolve()}else a.m.reject("empty");a.classList.add("recommendations-rendered")}else a.j.resolve()})};_ds.w(k1,_ds.R);k1.prototype.connectedCallback=function(){l1(this)};k1.prototype.disconnectedCallback=function(){_ds.Rl(this.eventHandler);this.j.reject("Disconnected")};
k1.prototype.render=function(a){if(this.isConnected){_ds.zl(this,j1,{Gd:a.hc()});a=Array.from(this.querySelectorAll(".significatio-date"));a=_ds.r(a);for(var b=a.next();!b.done;b=a.next()){b=b.value;var c=b.getAttribute("date");try{b.textContent=(new Date(1E3*Number(c))).toLocaleDateString("default",{month:"short",year:"numeric",day:"numeric",timeZone:this.timeZone})}catch(d){}}}};try{window.customElements.define("devsite-recommendations-sidebar",k1)}catch(a){console.warn("Unrecognized DevSite custom element - DevsiteRecommendationsSidebar",a)};})(_ds_www);
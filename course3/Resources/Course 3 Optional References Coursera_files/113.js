(window.webpackJsonp=window.webpackJsonp||[]).push([[113,118],{"+871":function(module,e,t){"use strict";t.r(e),t.d(e,"PracticeQuizAttemptPage",function(){return z}),t.d(e,"withRedirectToCover",function(){return A});var n=t("VbXa"),a=t.n(n),r=t("q1tI"),i=t.n(r),s=t("MnCE"),o=t("EqTq"),u=t("m10x"),c=t("+LJP"),d=t("usGt"),m=t("sQ/U"),l=t("JEIr"),p=t("oe9u"),f=t("/oRK"),b=t("XFHP"),S=t("22Sa"),g=t("eK4/"),I=t("Gpyo"),v=t("xmEj"),y=t("RH4a"),h=t("zaiP"),D=t("wIYG"),P=t("4QSv"),E=t("VtNW"),R=t.n(E),O=t("3AUy"),j=t.n(O),Q=Object(o.a)("PracticeQuizAttemptPage"),z=function(e){function PracticeQuizAttemptPage(){return e.apply(this,arguments)||this}a()(PracticeQuizAttemptPage,e);var t=PracticeQuizAttemptPage.prototype;return t.componentDidMount=function componentDidMount(){var e=this.props,t=e.shouldRedirectToCover,n=e.redirectToCover;t&&n()},t.render=function render(){var e=this.props,t=e.redirectToCover,n=e.redirectToNextItem,a=e.addRedirectToCoverToRouteParams,r=e.examSessionId;return i.a.createElement(h.a,null,function(e){var s=e.item;if(!s)return null;return i.a.createElement(S.a,{onClose:t,ariaLabel:R()("Practice Quiz"),backbuttonAriaLabel:R()("Back to the Course"),headerLeft:i.a.createElement(p.a,{headerText:s.name,itemTypeText:R()("Practice Quiz"),timeCommitment:s.timeCommitment}),headerRight:i.a.createElement(f.a,{deadline:s.deadline}),topBanner:i.a.createElement(l.a,{slug:s.courseSlug,itemId:s.id,userId:m.a.get().id},function(e){var a=e.bestEvaluation;return i.a.createElement(I.b,{slug:s.courseSlug,itemId:s.id,userId:m.a.get().id,fetchPolicy:"network-only",examSessionId:r},function(e){var r=e.isSubmitted,o,u=(s.contentSummary&&"quiz"===s.contentSummary.typeName&&s.contentSummary.definition||{}).passingFraction,c=a||{},d=c.score,m=void 0===d?0:d,l=c.maxScore,p=void 0===l?0:l,f=a?p&&m/p:void 0,b=f?f>=u:void 0;if(r&&u&&"number"==typeof f)return i.a.createElement(P.a,{passingFraction:u,itemGrade:{grade:f,isPassed:b||!1,isOverridden:!1,latePenaltyRatio:null},onKeepLearningClick:n,onTryAgainClick:t});return null})})},i.a.createElement(I.b,{slug:s.courseSlug,itemId:s.id,userId:m.a.get().id,fetchPolicy:"network-only",examSessionId:r},function(e){var t=e.loading,n=e.practiceQuizFormData,r=e.sessionId,o=e.nextSubmissionDraftId,c=e.totalPoints,d=e.isSubmitted,m=e.hasDraft;if(t||!n)return i.a.createElement(D.a,null);var l=n.map(function(e){return e.prompt.id});return i.a.createElement(y.a,{itemId:s.id,courseId:s.courseId},function(e){var t=e.stepState,p=e.setStepState;return i.a.createElement("div",{className:Q()},i.a.createElement("div",{className:Q("header")},i.a.createElement(u.h,{tag:"h2"},s.name),i.a.createElement(u.a,{rootClassName:Q("points")},i.a.createElement(u.l,{tag:"span"},R()("Total points #{totalPoints}",{totalPoints:c})))),i.a.createElement("div",{className:Q("prompts")},n.map(function(e,n){return i.a.createElement(g.a,{key:e.prompt.id,quizQuestion:e,index:n,isReadOnly:!!d,isDisabled:!!(t||{}).isSaving||!!(t||{}).isSubmitting})})),!d&&i.a.createElement(v.a,{ids:l,sessionId:r,nextSubmissionDraftId:o},function(e){var n=e.saveDraft,r=e.autoSaveDraft,o=e.submitDraft;return i.a.createElement(b.a,{itemId:s.id,courseId:s.courseId,saveDraft:n,autoSaveDraft:r,submitDraft:function submitDraft(){return o?o().then(function(){a()}):Promise.reject()},hasDraft:m,stepState:t,setStepState:p})}))})}))})},PracticeQuizAttemptPage}(i.a.Component),A=Object(s.compose)(d.a,Object(c.a)(function(e,t){var n=t.nextItemUrl,a=void 0===n?"":n,r=t.refetchPracticeQuizCoverPageData,i=t.refreshProgress,s=function redirectToCover(){r&&r(),e.push({name:"practice-quiz-cover",params:e.params,query:e.location.query}),i()};return{redirectToCover:s,redirectToNextItem:function redirectToNextItem(){a?(e.push(a),i()):s()},addRedirectToCoverToRouteParams:function addRedirectToCoverToRouteParams(){e.push({name:"practice-quiz-attempt",params:e.params,query:{redirectToCover:!0}})},shouldRedirectToCover:e.location.query.redirectToCover}}));e.default=A(z)},"3AUy":function(module,exports,e){var t=e("aTjV"),n;"string"==typeof t&&(t=[[module.i,t,""]]);var a={transform:void 0},r=e("aET+")(t,a);t.locals&&(module.exports=t.locals)},Gpyo:function(module,e,t){"use strict";t.d(e,"a",function(){return p}),t.d(e,"c",function(){return S});var n=t("lSNA"),a=t.n(n),r=t("VkAN"),i=t.n(r),s=t("q1tI"),o=t.n(s),u=t("lTCR"),c=t.n(u),d=t("oJmH"),m=t.n(d),l=t("TUhR");function ownKeys(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);t&&(a=a.filter(function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable})),n.push.apply(n,a)}return n}function _objectSpread(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?ownKeys(Object(n),!0).forEach(function(t){a()(e,t,n[t])}):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):ownKeys(Object(n)).forEach(function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))})}return e}function _templateObject2(){var e=i()(['\n  query practiceQuizDataQuery(\n    $itemId: String!\n    $userId: Int!\n    $slug: String!\n    $sessionId: String!\n    $body: String!\n    $skipFetchingResponses: Boolean!\n  ) {\n    practiceQuizQuestionData(itemId: $itemId, userId: $userId, slug: $slug, sessionId: $sessionId, body: $body)\n      @rest(\n        type: "RestPracticeQuizQuestionGetStateData"\n        path: "opencourse.v1/user/{args.userId}/course/{args.slug}/item/{args.itemId}/quiz/session/{args.sessionId}/action/getState?autoEnroll=false"\n        method: "POST"\n        bodyKey: "body"\n      ) {\n      contentResponseBody @type(name: "RestPracticeQuizQuestionDataResponseBody") {\n        return @type(name: "RestContentResponseBodyReturnObject") {\n          questions\n          nextSubmissionDraftId\n        }\n      }\n    }\n    practiceQuizQuestionResponses(itemId: $itemId, userId: $userId, slug: $slug, sessionId: $sessionId, body: $body)\n      @skip(if: $skipFetchingResponses)\n      @rest(\n        type: "RestPracticeQuizLatestSubmissionDraftResponse"\n        path: "opencourse.v1/user/{args.userId}/course/{args.slug}/item/{args.itemId}/quiz/session/{args.sessionId}/action/getLatestSubmissionDraft?autoEnroll=false"\n        method: "POST"\n        bodyKey: "body"\n      ) {\n      contentResponseBody @type(name: "RestPracticeQuizQuestionDataResponseBody") {\n        return @type(name: "RestContentResponseBodyReturnObject") {\n          draft\n        }\n      }\n    }\n  }\n']);return _templateObject2=function _templateObject2(){return e},e}function _templateObject(){var e=i()(['\n  query practiceQuizSessionMetaDataQuery($itemId: String!, $userId: Int!, $slug: String!, $body: String!) {\n    practiceQuizSessionMetaData(body: $body, itemId: $itemId, userId: $userId, slug: $slug)\n      @rest(\n        type: "RestPracticeQuizSessionMetadata"\n        path: "opencourse.v1/user/{args.userId}/course/{args.slug}/item/{args.itemId}/quiz/session"\n        method: "POST"\n        bodyKey: "body"\n      ) {\n      contentResponseBody @type(name: "RestPracticeQuizSessionDataResponseBody")\n    }\n  }\n']);return _templateObject=function _templateObject(){return e},e}var p={contentRequestBody:{argument:[]}},f={contentRequestBody:[]},b=c()(_templateObject()),S=c()(_templateObject2()),g=function PracticeQuizData(e){var t=e.slug,n=e.itemId,a=e.userId,r=e.fetchPolicy,i=void 0===r?"cache-first":r,s=e.examSessionId,u=e.children,c={slug:t,itemId:n,userId:a};return o.a.createElement(d.Query,{query:b,variables:_objectSpread(_objectSpread({},c),{},{body:f}),fetchPolicy:i,skip:!!s},function(e){var t=e.loading,n=e.data,a=e.refetch;if(t&&!s)return u({loading:t});var r=s||((((n||{}).practiceQuizSessionMetaData||{}).contentResponseBody||{}).session||{}).id;return o.a.createElement(d.Query,{query:S,variables:_objectSpread(_objectSpread({},c),{},{sessionId:r,body:p,skipFetchingResponses:!!s}),fetchPolicy:i},function(e){var t=e.loading,n=e.data,i=e.refetch,o=e.client;if(t)return u({loading:t});var c=(((n||{}).practiceQuizQuestionData||{}).contentResponseBody||{}).return,d=c.questions,m=c.nextSubmissionDraftId,p,f,b,S=((((((n||{}).practiceQuizQuestionResponses||{}).contentResponseBody||{}).return||{}).draft||{}).input||{}).responses,g=Object(l.a)(d,S,!!s),I=g.reduce(function(e,t){return e+t.prompt.weightedScoring.maxScore},0),v=!d[0].isSubmitAllowed,y,h;return u({loading:t,practiceQuizFormData:g,sessionId:r,totalPoints:I,isSubmitted:v,nextSubmissionDraftId:m,hasDraft:!!S,refetchPracticeQuizData:function refetchPracticeQuizData(){return a().then(function(){return i()})},client:o})})})};e.b=g},XFHP:function(module,e,t){"use strict";var n=t("lSNA"),a=t.n(n),r=t("VbXa"),i=t.n(r),s=t("q1tI"),o=t.n(s),u=t("XBa+"),c=t("uhOI"),d=t("d3Ej"),m=t.n(d),l=t("EqTq"),p=t("HOoY"),f=t("zaiP"),b=t("scbn"),S=t("qJwm"),g=t("F9Z8"),I=t("CnKM"),v=t("8ec0"),y=t("KvdX"),h=t("rQpo"),D=t("Cqp/"),P=t("lEBL"),E=t.n(P),R=t("PB6g");function ownKeys(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);t&&(a=a.filter(function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable})),n.push.apply(n,a)}return n}function _objectSpread(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?ownKeys(Object(n),!0).forEach(function(t){a()(e,t,n[t])}):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):ownKeys(Object(n)).forEach(function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))})}return e}var O=Object(l.a)("SubmissionControls"),j=function(e){function SubmissionControls(){for(var t,n=arguments.length,a=new Array(n),r=0;r<n;r++)a[r]=arguments[r];return(t=e.call.apply(e,[this].concat(a))||this).state={canSubmit:!1,announcement:""},t.onSubmitClick=function(){var e=t.props,n=e.hasUnfilledResponses,a=e.updateAndSubmitDraft,r=e.showModal,i=function trackedUpdateAndSubmitDraft(){return a&&a().then(function(){p.a.trackComponent({namespace:{app:"open_course",page:"item_page"}},{},"submit","quiz")})};n?r({type:y.a.unansweredQuestions,props:{onPrimaryButtonClick:i}}):a&&(t.setAnnouncement(m()("Submitting assignment")),i())},t.enableSubmit=function(){t.setState({canSubmit:!0})},t.disableSubmit=function(){t.setState({canSubmit:!1})},t.autoSubmit=function(){var e=t.props,n=e.showModal,a=e.hideModal,r=e.submitLatestSubmissionDraft;n({type:y.a.timeExpiredModal,props:{onPrimaryButtonClick:a}}),r&&r()},t.startAutoSubmitTimeout=function(){var e=t.props.expiresAt;"number"==typeof e&&(t.autoSubmitTimeout=window.setTimeout(t.autoSubmit,e-Date.now()))},t.setAnnouncement=function(e){t.setState({announcement:e},function(){setTimeout(function(){t.setState({announcement:""})},150)})},t.updateDraft=function(){var e=t.props.updateDraft;e&&(t.setAnnouncement(m()("Saving draft submission")),e().then(function(){t.setAnnouncement(m()("Draft submission saved successfully"))}))},t}i()(SubmissionControls,e);var t=SubmissionControls.prototype;return t.componentDidMount=function componentDidMount(){var e=this.props,t=e.hasDraft,n=e.autoSaveDraft,a=e.submitLatestSubmissionDraft;!t&&n&&n(),a&&this.startAutoSubmitTimeout()},t.componentDidUpdate=function componentDidUpdate(e){var t=this.props.expiresAt;e.expiresAt!==t&&this.autoSubmitTimeout&&(clearTimeout(this.autoSubmitTimeout),this.autoSubmitTimeout=null,this.startAutoSubmitTimeout())},t.componentWillUnmount=function componentWillUnmount(){this.autoSubmitTimeout&&clearTimeout(this.autoSubmitTimeout)},t.render=function render(){var e=this,t=this.props,n=t.stepState,a=n.isSaving,r=n.isSubmitting,i=n.isAutoSaving,s=t.stepState,d=t.setStepState,l=t.updateDraft,p=t.updateAndSubmitDraft,I=this.state,v=I.canSubmit,y=I.announcement,P=a||i||r;return o.a.createElement(f.a,null,function(t){var n=t.item;if(n&&n.isPremiumGradingLocked)return o.a.createElement("div",{className:O("upgrade-button")},o.a.createElement(h.a,{courseId:n.courseId}));return o.a.createElement("div",{className:O()},o.a.createElement("div",{className:O("honor-code-and-feedback")},o.a.createElement(g.a,{onAgreementComplete:e.enableSubmit,onAgreementIncomplete:e.disableSubmit}),o.a.createElement(b.a,{computedItem:n,itemFeedbackType:S.c.Quiz})),o.a.createElement("div",{className:O("buttons")},l&&o.a.createElement(u.a,{onClick:e.updateDraft,disabled:P,apiStatus:a||i?c.c:c.a,type:"secondary",rootClassName:O("button"),apiStatusAttributesConfig:{label:{API_BEFORE_SEND:m()("Save"),API_IN_PROGRESS:m()("Saving...")}}}),p&&o.a.createElement(u.a,{onClick:e.onSubmitClick,disabled:P||!v,apiStatus:r?c.c:c.a,type:"primary",rootClassName:O("button"),apiStatusAttributesConfig:{label:{API_BEFORE_SEND:m()("Submit"),API_IN_PROGRESS:m()("Submitting…")}}})),o.a.createElement(R.b,{tagName:"span",role:"region","aria-live":"assertive","aria-atomic":!0},y&&o.a.createElement("span",null,y)),o.a.createElement(D.a,{stepState:s,setStepState:d}))})},SubmissionControls}(o.a.Component),Q=function SubmissionControlsContainer(e){var t=e.saveDraft,n=e.submitDraft,a=e.hasTimer,r=e.itemId,i=e.isSubmitted;if(!t||!n)return null;if(!a)return o.a.createElement(y.b,null,function(a){var r=a.showModal,s=a.hideModal;return i?null:o.a.createElement(j,_objectSpread(_objectSpread({},e),{},{updateDraft:t,updateAndSubmitDraft:n,showModal:r,hideModal:s}))});return o.a.createElement(I.a,{id:Object(v.a)(r)},function(a){var r=a.expiresAt;return o.a.createElement(y.b,null,function(a){var s=a.showModal,u=a.hideModal;return i?null:o.a.createElement(j,_objectSpread(_objectSpread({},e),{},{updateDraft:t,updateAndSubmitDraft:n,showModal:s,hideModal:u,expiresAt:r}))})})};e.a=Q},"a7+h":function(module,e,t){"use strict";t.d(e,"a",function(){return b});var n=t("VkAN"),a=t.n(n),r=t("q1tI"),i=t.n(r),s=t("lTCR"),o=t.n(s),u=t("oJmH"),c=t.n(u),d=t("kqcj");function _templateObject2(){var e=a()(['\n  mutation practiceQuizSubmitResponseActionMutation(\n    $itemId: String!\n    $userId: Int!\n    $slug: String!\n    $sessionId: String!\n    $body: String!\n  ) {\n    action(itemId: $itemId, userId: $userId, slug: $slug, sessionId: $sessionId, body: $body)\n      @rest(\n        type: "PracticeQuizSubmitResponseData"\n        path: "opencourse.v1/user/{args.userId}/course/{args.slug}/item/{args.itemId}/quiz/session/{args.sessionId}/action/submitResponses?autoEnroll=false"\n        method: "POST"\n        bodyKey: "body"\n      ) {\n      contentResponseBody @type(name: "RestPracticeQuizQuestionDataResponseBody") {\n        return @type(name: "RestContentResponseBodyReturnObject") {\n          nextSubmissionDraftId\n          evaluation\n          questions\n        }\n      }\n    }\n  }\n']);return _templateObject2=function _templateObject2(){return e},e}function _templateObject(){var e=a()(['\n  mutation practiceQuizSaveSubmissionDraftActionMutation(\n    $itemId: String!\n    $userId: Int!\n    $slug: String!\n    $sessionId: String!\n    $body: String!\n    $additionalParams: String!\n  ) {\n    action(\n      itemId: $itemId\n      userId: $userId\n      slug: $slug\n      sessionId: $sessionId\n      body: $body\n      additionalParams: $additionalParams\n    )\n      @rest(\n        type: "PracticeQuizSaveSubmissionDraftData"\n        path: "opencourse.v1/user/{args.userId}/course/{args.slug}/item/{args.itemId}/quiz/session/{args.sessionId}/action/saveSubmissionDraft?autoEnroll=false{args.additionalParams}"\n        method: "POST"\n        bodyKey: "body"\n      ) {\n      contentResponseBody @type(name: "RestPracticeQuizQuestionDataResponseBody") {\n        return @type(name: "RestContentResponseBodyReturnObject") {\n          nextSubmissionDraftId\n          questions @skip(if: $skipQuestionsField)\n          evaluation\n        }\n      }\n    }\n  }\n']);return _templateObject=function _templateObject(){return e},e}var m=o()(_templateObject()),l=o()(_templateObject2()),p=function SaveDraftMutation(e){var t=e||{},n=t.changedResponses,a=t.nextSubmissionDraftId,r=t.sessionId,s=t.userId,o=t.itemId,c=t.slug;return i.a.createElement(u.Mutation,{mutation:m,update:Object(d.a)(e)},function(t){var i={contentRequestBody:{argument:{id:a,input:{responses:n}}}},u=function saveDraft(){return t({variables:{body:i,sessionId:r,userId:s,itemId:o,slug:c,skipQuestionsField:!0,additionalParams:""}})},d=function autoSaveDraft(){return t({variables:{body:i,sessionId:r,userId:s,itemId:o,slug:c,skipQuestionsField:!0,additionalParams:"&isAutoSaving=true"}})};return e.children({saveDraft:u,autoSaveDraft:d})})},f=function SubmitDraftMutation(e){var t=e||{},n=t.changedResponses,a=t.sessionId,r=t.userId,s=t.itemId,o=t.slug;return i.a.createElement(u.Mutation,{mutation:l,update:Object(d.a)(e)},function(t){var i={contentRequestBody:{argument:{responses:n}}},u=function submitDraft(){return t({variables:{body:i,sessionId:a,userId:r,itemId:s,slug:o}})};return e.children({submitDraft:u})})},b=function PracticeQuizMutations(e){var t=e.ids,n=e.sessionId,a=e.itemId,r=e.courseId,s=e.changedResponses,o=e.children,u=e.slug,c=e.userId,d=e.nextSubmissionDraftId;return i.a.createElement(p,{changedResponses:s,nextSubmissionDraftId:d,sessionId:n,userId:c,itemId:a,slug:u},function(e){var t=e.saveDraft,r=e.autoSaveDraft;return i.a.createElement(f,{changedResponses:s,sessionId:n,userId:c,itemId:a,slug:u},function(e){var n=e.submitDraft;return o({saveDraft:t,autoSaveDraft:r,submitDraft:n})})})},S=b},aTjV:function(module,exports,e){},glp6:function(module,e,t){"use strict";t.r(e);var n=t("q1tI"),a=t.n(n),r=t("sQ/U"),i=t("zaiP"),s=t("JEIr"),o=t("+871"),u=function SubmittedPracticeQuizAttemptPage(e){var t=e.nextItemUrl;return a.a.createElement(i.a,null,function(e){var n=e.item;if(!n||!n.contentSummary)return null;return a.a.createElement(s.a,{slug:n.courseSlug,itemId:n.id,userId:r.a.get().id},function(e){var r=e.loading,i=e.lastSessionId;if(!n||!n.contentSummary||r)return null;return a.a.createElement(o.default,{examSessionId:i,nextItemUrl:t})})})};e.default=u},kqcj:function(module,e,t){"use strict";var n=t("Gpyo"),a=function practiceQuizActionMutationUpdate(e){return function(t,a){var r=a.data,i=e.itemId,s=e.userId,o=e.slug,u=e.sessionId,c=(((r||{}).action||{}).contentResponseBody||{}).return||null,d=t.readQuery({query:n.c,variables:{itemId:i,userId:s,slug:o,sessionId:u,skipFetchingResponses:!1,body:n.a}}),m=(((d||{}).practiceQuizQuestionData||{}).contentResponseBody||{}).return||null;m&&(m.nextSubmissionDraftId=(c||{}).nextSubmissionDraftId||null,m.questions=(c||{}).questions||m.questions||null,m.evaluation=(c||{}).evaluation||null,t.writeQuery({query:n.c,variables:{itemId:i,userId:s,slug:o,sessionId:u,skipFetchingResponses:!1,body:n.a},data:d}))}};e.a=a},lEBL:function(module,exports,e){var t=e("lWSP"),n;"string"==typeof t&&(t=[[module.i,t,""]]);var a={transform:void 0},r=e("aET+")(t,a);t.locals&&(module.exports=t.locals)},lWSP:function(module,exports,e){},xmEj:function(module,e,t){"use strict";var n=t("q1tI"),a=t.n(n),r=t("agp5"),i=t("RH4a"),s=t("0n3a"),o=t("zaiP"),u=t("JEIr"),c=t("sQ/U"),d=t("wjH1"),m=t("a7+h"),l=function PracticeQuizActions(e){var t=e.ids,n=e.sessionId,l=e.children,p=e.nextSubmissionDraftId;return a.a.createElement(o.a,null,function(e){var o=e.item;if(!o)return null;var f=o.courseId,b=o.id,S=o.courseSlug;return a.a.createElement(u.a,{userId:c.a.get().id,itemId:b,slug:S},function(e){var u=e.refetch;return a.a.createElement(r.a,{ids:t},function(e){if(!e||!u)return l({});return a.a.createElement(i.a,{itemId:b,courseId:f},function(r){var i=r.stepState,g=r.setStepState,I=e.map(function(e){return{questionInstance:e.id,response:((e.response||{}).definition||{}).value}});return a.a.createElement(m.a,{ids:t,sessionId:n,itemId:b,courseId:f,slug:S,userId:c.a.get().id,changedResponses:I,nextSubmissionDraftId:p},function(t){var n=t.saveDraft,r=t.submitDraft,c=function saveDraft(){if(!i.isSaving&&!i.isSubmitting&&p)return g({isSaving:!0}),n().then(function(){return g({isSaving:!1})}).catch(function(e){return Object(d.a)(e,g)});return Promise.reject()},m=function autoSaveDraft(){if(!i.isAutoSaving&&!i.isSubmitting&&p)return g({isAutoSaving:!0}),n().then(function(){return g({isAutoSaving:!1})}).catch(function(e){return Object(d.a)(e,g)});return Promise.reject()},f=function refetchItemAndEvaluations(){return u().then(function(){return o.refetch()}).catch(function(e){return Object(d.a)(e,g)})},b=function submitDraft(){if(!i.isSubmitting)return g({isSubmitting:!0}),r().then(function(){return g({isSubmitting:!1})}).then(f).catch(function(e){return Object(d.a)(e,g)});return Promise.reject()};return a.a.createElement(s.a,{stepState:i,saveDraft:m,changedResponses:e},function(){return l({saveDraft:c,autoSaveDraft:m,submitDraft:b})})})})})})})};e.a=l}}]);
//# sourceMappingURL=113.31789e97e1e58522869d.js.map
/*! instantDataScraper - 2018-02-26 */
//extension "instantDataScraper".

// 拦截请求并修改请求头
chrome.webRequest.onBeforeSendHeaders.addListener(
    function(details) {
      details.requestHeaders.push({
        name: "Access-Control-Allow-Origin",
        value: "*"
      });
      return { requestHeaders: details.requestHeaders };
    },
    { urls: ["<all_urls>"] },
    ["blocking", "requestHeaders"]
  );
  
  // 拦截响应并修改响应头
  chrome.webRequest.onHeadersReceived.addListener(
    function(details) {
      details.responseHeaders.push({
        name: "Access-Control-Allow-Origin",
        value: "*"
      });
      return { responseHeaders: details.responseHeaders };
    },
    { urls: ["<all_urls>"] },
    ["blocking", "responseHeaders"]
  );
  
chrome.browserAction.onClicked.addListener(function(a){chrome.windows.getCurrent(function(a){parentWindowId=a.id});window.open(chrome.extension.getURL("popup.html?tabid="+encodeURIComponent(a.id)+"&url="+encodeURIComponent(a.url)),"Table Scraper","toolbar=0,scrollbars=0,location=0,statusbar=0,menubar=0,resizable=1,width=720,height=650")});

document.getElementById('sendButton').addEventListener('click', sendMessage);
document.getElementById('sendurlButton').addEventListener('click', sendurlMessage);

// 发送用户输入的消息和 PDF 文件到 Flask 后端
async function sendMessage() {
  const userInput = document.getElementById('userInput').value;

  displayInput(userInput)

  const formData = new FormData();
  formData.append('user_input', userInput);

  try {
    const response = await fetch('http://localhost:5000/process_pdf', { // Flask 服务地址
      method: 'POST',
      body: formData,
    });

    const data = await response.json();
    displayResponse(data.response);
    document.getElementById('userInput').value = '';  // 清空输入框
  } catch (error) {
    console.error('Error:', error);
    displayResponse('Error: Unable to fetch response from server.');
  }
}

async function sendurlMessage() {
  const pdfUrl = document.getElementById('urlInput').value;
  if (pdfUrl) {
    await downloadFilesFromUrl(pdfUrl); // 調用下載函數
  } else {
    alert('Please enter a valid URL.'); // 如果未輸入 URL，提示用戶
  }

  await new Promise(resolve => setTimeout(resolve, 5000));

  const formData = new FormData();
  formData.append('pdf_url',pdfUrl)
  try {
    const response = await fetch('http://localhost:5000/enter_url', { // Flask 服务地址
      method: 'POST',
      body: formData,
    });

    displayResponse("已讀取成功");
  } catch (error) {
    console.error('Error:', error);
    displayResponse('Error: Unable to fetch response from server.');
  }
}

function downloadFilesFromUrl(url) {
  // 使用 jQuery 的 Ajax 加載目標 URL 的 HTML 內容
  const proxyUrl = 'https://cors-anywhere.herokuapp.com/';
  $.ajax({
      url: proxyUrl+url,
      method: 'GET',
      success: function(response) {
      // 將返回的 HTML 內容轉換為 jQuery 對象
      var htmlContent = $(response);

      // 選擇所有可能的文件鏈接（假設以常見文件類型結尾）
      var fileLinks = htmlContent.find("a[href$='.pdf'], a[href$='.doc'], a[href$='.docx'], a[href$='.xls'], a[href$='.xlsx'], a[href$='.zip'], a[href$='.odt'], a[href$='.ods']");
      // 遍歷每個文件鏈接
      fileLinks.each(function() {
          var link = $(this).attr('href'); // 獲取文件的 href 屬性
          var fileName = link.split('/').pop(); // 從 URL 提取文件名

          // 如果鏈接是相對路徑，則轉換為絕對路徑
          if (!link.startsWith('http')) {
          var absoluteLink = new URL(link, url).href;
          link = absoluteLink;
          }

          // 打印正在抓取的文件URL
          console.log('Downloading file from URL:', link);

          // 使用 Fetch API 下載文件並使用 FileSaver.js 保存文件
          fetch(proxyUrl+link)
          .then(response => response.blob()) // 將響應轉換為 Blob
          .then(blob => {
              console.log('Downloaded:', fileName); // 顯示已下載的文件名
              saveAs(blob, fileName); // 使用 FileSaver.js 的 saveAs 函數保存文件
          })
          .catch(error => console.error('Error downloading file:', error));
      });
    },
      error: function(error) {
      console.error('Error loading the page:', error);
      }
  });
}

//不用動
function displayInput(userInput) {
  const chatbox = document.getElementById('chatbox');

  const userWrapper = document.createElement('div');
  userWrapper.classList.add('message-wrapper', 'user-message-wrapper');

  const userMessage = document.createElement('div');
  userMessage.classList.add('message', 'user-message');
  userMessage.innerText = userInput;

  userWrapper.appendChild(userMessage);

  chatbox.appendChild(userWrapper);

  chatbox.scrollTop = chatbox.scrollHeight;
}
//不用動
function displayResponse(response) {
  const chatbox = document.getElementById('chatbox');

  const botWrapper = document.createElement('div');
  botWrapper.classList.add('message-wrapper', 'bot-message-wrapper');

  const botMessage = document.createElement('div');
  botMessage.classList.add('message', 'bot-message');
  botMessage.innerText = response;

  botWrapper.appendChild(botMessage);

  chatbox.appendChild(botWrapper);

  chatbox.scrollTop = chatbox.scrollHeight;
}



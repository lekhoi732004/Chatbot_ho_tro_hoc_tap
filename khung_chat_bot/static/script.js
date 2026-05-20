document.addEventListener("DOMContentLoaded", () => {
  const chatBox = document.getElementById("chat-box");
  const userInput = document.getElementById("user-input");
  const sendBtn = document.getElementById("send-btn");
  const toggleSidebarBtn = document.getElementById("toggle-sidebar");
  const sidebar = document.getElementById("history-sidebar");
  const historyList = document.getElementById("history-list");
  const newChatBtns = [
    document.getElementById("new-chat-btn"),
    document.getElementById("new-chat-top-btn"),
  ];
  const settingsBtn = document.getElementById("settings-btn");
  const settingsModal = document.getElementById("settings-modal");
  const closeSettings = document.getElementById("close-settings");
  const themeToggle = document.getElementById("theme-toggle");
  const attachBtn = document.getElementById("attach-btn");
  const attachMenu = document.getElementById("attach-menu");

  let chatSessions = JSON.parse(localStorage.getItem("eduAgentChats")) || [];
  let currentSessionId = null;
  let isSending = false;

  init();

  function init() {
    renderSidebar();
    if (chatSessions.length > 0) {
      loadChatSession(chatSessions[0].id);
    } else {
      startNewChat();
    }

    if (localStorage.getItem("eduTheme") === "light") {
      document.body.classList.add("light-mode");
      themeToggle.checked = true;
    }
  }

  function saveToLocalStorage() {
    localStorage.setItem("eduAgentChats", JSON.stringify(chatSessions));
  }

  function getCurrentSession() {
    return chatSessions.find((session) => session.id === currentSessionId);
  }

  function startNewChat() {
    currentSessionId = null;
    chatBox.innerHTML = "";
    appendMessage(
      "bot",
      "Xin chào. Mình là Edu Chatbot. Bạn cần hỗ trợ bài tập hay kiến thức gì hôm nay?"
    );
    markActiveSession();
  }

  function createSession(firstMessage) {
    const session = {
      id: `chat_${Date.now()}`,
      title: firstMessage.substring(0, 35) + (firstMessage.length > 35 ? "..." : ""),
      messages: [],
    };
    chatSessions.unshift(session);
    currentSessionId = session.id;
    saveToLocalStorage();
    renderSidebar();
    return session;
  }

  async function sendMessage() {
    const text = userInput.value.trim();
    if (!text || isSending) return;

    isSending = true;
    sendBtn.disabled = true;

    let session = getCurrentSession();
    if (!session) {
      session = createSession(text);
      chatBox.innerHTML = "";
    }

    appendMessage("user", text);
    session.messages.push({ sender: "user", text });
    saveToLocalStorage();

    userInput.value = "";
    userInput.style.height = "auto";

    const thinkingNode = appendMessage("bot", "Đang suy nghĩ...");

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          session_id: currentSessionId,
        }),
      });

      const data = await response.json();
      if (!response.ok || data.status !== "success") {
        throw new Error(data.reply || "Lỗi máy chủ");
      }

      const reply = data.reply || "[Không có câu trả lời]";
      updateMessage(thinkingNode, reply);
      session.messages.push({
        sender: "bot",
        text: reply,
        metadata: data.metadata || {},
      });
      saveToLocalStorage();
    } catch (error) {
      const message = `Xin lỗi, không thể kết nối với máy chủ AI. ${error.message}`;
      updateMessage(thinkingNode, message);
      session.messages.push({ sender: "bot", text: message });
      saveToLocalStorage();
    } finally {
      isSending = false;
      sendBtn.disabled = false;
      userInput.focus();
    }
  }

  function appendMessage(sender, text) {
    const messageDiv = document.createElement("div");
    messageDiv.classList.add("message", sender);

    const contentDiv = document.createElement("div");
    contentDiv.classList.add("message-content");
    contentDiv.textContent = text;

    messageDiv.appendChild(contentDiv);
    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
    return messageDiv;
  }

  function updateMessage(messageNode, text) {
    const content = messageNode.querySelector(".message-content");
    if (content) content.textContent = text;
    chatBox.scrollTop = chatBox.scrollHeight;
  }

  function loadChatSession(sessionId) {
    currentSessionId = sessionId;
    const session = getCurrentSession();
    chatBox.innerHTML = "";

    if (session && session.messages.length > 0) {
      session.messages.forEach((msg) => appendMessage(msg.sender, msg.text));
    } else {
      appendMessage(
        "bot",
        "Xin chào. Mình là Edu Chatbot. Bạn cần hỗ trợ bài tập hay kiến thức gì hôm nay?"
      );
    }

    markActiveSession();
  }

  function renderSidebar() {
    historyList.innerHTML = "";

    chatSessions.forEach((session) => {
      const item = document.createElement("div");
      item.className = "history-item";
      item.dataset.id = session.id;
      item.innerHTML = `
        <i class="far fa-message item-icon"></i>
        <span class="chat-title"></span>
        <div class="menu-container">
          <button class="more-btn" title="Tùy chọn"><i class="fas fa-ellipsis-vertical"></i></button>
          <div class="dropdown-menu">
            <div class="dropdown-item rename-btn"><i class="fas fa-pen"></i> Đổi tên</div>
            <div class="dropdown-item delete-btn"><i class="fas fa-trash"></i> Xóa</div>
          </div>
        </div>
      `;
      item.querySelector(".chat-title").textContent = session.title;
      historyList.appendChild(item);
    });

    markActiveSession();
  }

  function markActiveSession() {
    document.querySelectorAll(".history-item").forEach((item) => {
      item.style.backgroundColor =
        item.dataset.id === currentSessionId ? "var(--bg-hover)" : "transparent";
    });
  }

  attachBtn.addEventListener("click", (event) => {
    event.stopPropagation();
    attachMenu.classList.toggle("show");
  });

  document.querySelectorAll(".attach-item").forEach((item) => {
    item.addEventListener("click", () => {
      alert("Chức năng đính kèm đang được phát triển.");
      attachMenu.classList.remove("show");
    });
  });

  document.addEventListener("click", (event) => {
    if (!event.target.closest(".menu-container")) {
      document.querySelectorAll(".dropdown-menu").forEach((menu) => {
        menu.classList.remove("show");
      });
    }
    if (!event.target.closest(".attachment-wrapper")) {
      attachMenu.classList.remove("show");
    }
    if (event.target === settingsModal) {
      settingsModal.style.display = "none";
    }
  });

  settingsBtn.addEventListener("click", () => {
    settingsModal.style.display = "flex";
  });

  closeSettings.addEventListener("click", () => {
    settingsModal.style.display = "none";
  });

  themeToggle.addEventListener("change", (event) => {
    if (event.target.checked) {
      document.body.classList.add("light-mode");
      localStorage.setItem("eduTheme", "light");
    } else {
      document.body.classList.remove("light-mode");
      localStorage.setItem("eduTheme", "dark");
    }
  });

  toggleSidebarBtn.addEventListener("click", () => {
    sidebar.classList.toggle("hidden");
  });

  newChatBtns.forEach((btn) => {
    if (btn) btn.addEventListener("click", startNewChat);
  });

  historyList.addEventListener("click", (event) => {
    const historyItem = event.target.closest(".history-item");
    if (!historyItem) return;

    const sessionId = historyItem.dataset.id;

    if (event.target.closest(".more-btn")) {
      event.stopPropagation();
      document.querySelectorAll(".dropdown-menu").forEach((menu) => {
        menu.classList.remove("show");
      });
      historyItem.querySelector(".dropdown-menu").classList.toggle("show");
      return;
    }

    if (event.target.closest(".rename-btn")) {
      event.stopPropagation();
      const session = chatSessions.find((item) => item.id === sessionId);
      const newTitle = prompt("Nhập tên cuộc trò chuyện mới:", session.title);
      if (newTitle) {
        session.title = newTitle;
        saveToLocalStorage();
        renderSidebar();
      }
      return;
    }

    if (event.target.closest(".delete-btn")) {
      event.stopPropagation();
      if (confirm("Bạn có chắc muốn xóa lịch sử này?")) {
        chatSessions = chatSessions.filter((item) => item.id !== sessionId);
        saveToLocalStorage();
        renderSidebar();
        if (currentSessionId === sessionId) startNewChat();
      }
      return;
    }

    loadChatSession(sessionId);
  });

  userInput.addEventListener("input", function () {
    this.style.height = "auto";
    this.style.height = `${Math.min(this.scrollHeight, 200)}px`;
  });

  userInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  });

  sendBtn.addEventListener("click", sendMessage);

  // ═══════════════════════════════════════════════════════════════════════
  // File Upload Handlers
  // ═══════════════════════════════════════════════════════════════════════

  const fileInput = document.getElementById("file-input");
  const uploadFileBtn = document.getElementById("upload-file-btn");

  uploadFileBtn.addEventListener("click", () => {
    fileInput.click();
    attachMenu.classList.remove("show");
  });

  async function uploadFile(file) {
    if (!file) return;

    if (file.size > 100 * 1024 * 1024) {
      appendMessage("bot", "❌ File quá lớn. Vui lòng chọn file nhỏ hơn 100MB.");
      return;
    }

    let session = getCurrentSession();
    if (!session) {
      session = createSession(file.name);
      chatBox.innerHTML = "";
    }

    const uploadMsg = appendMessage("bot", `📤 Đang tải lên "${file.name}"...`);
    isSending = true;
    sendBtn.disabled = true;

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("session_id", currentSessionId);

      const response = await fetch("/api/ingest", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (data.status === "success") {
        const successMsg = `✅ Tải tệp thành công!\n\n📄 Tệp: ${data.file_name}\n📏 Kích thước: ${(data.text_length / 1024).toFixed(2)} KB\n📋 Loại: ${data.file_type}\n📊 Số lượng: ${data.metadata.chunks_created || 0} phần\n\nBây giờ bạn có thể hỏi câu hỏi về nội dung tệp này.`;
        updateMessage(uploadMsg, successMsg);
        session.messages.push({
          sender: "bot",
          text: successMsg,
          metadata: data.metadata || {},
        });
      } else {
        const errorMsg = `❌ Lỗi tải tệp: ${data.error || "Lỗi không xác định"}`;
        updateMessage(uploadMsg, errorMsg);
        session.messages.push({ sender: "bot", text: errorMsg });
      }

      saveToLocalStorage();
    } catch (error) {
      const errorMsg = `❌ Lỗi kết nối: ${error.message}`;
      updateMessage(uploadMsg, errorMsg);
      session.messages.push({ sender: "bot", text: errorMsg });
      saveToLocalStorage();
    } finally {
      isSending = false;
      sendBtn.disabled = false;
      userInput.focus();
    }
  }

  fileInput.addEventListener("change", (event) => {
    const file = event.target.files[0];
    if (file) {
      uploadFile(file);
      fileInput.value = "";
    }
  });

});

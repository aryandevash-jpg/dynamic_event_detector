const detectResult = document.getElementById("detect-result");
const tweetInput = document.getElementById("tweet");

function showResult(node, html) {
  node.innerHTML = html;
  node.classList.remove("hidden");
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(error || `Request failed with status ${response.status}`);
  }

  return response.json();
}

document.getElementById("detect-button").addEventListener("click", async () => {
  const tweet = tweetInput.value.trim();

  if (!tweet) {
    showResult(detectResult, "<p class='status-message error'>INPUT REQUIRED.</p>");
    return;
  }

  showResult(detectResult, "<p class='status-message'>ANALYZING SIGNAL...</p>");

  try {
    const data = await postJson("/detect", { tweet });
    const prediction = data.prediction;
    
    // Build Top Matches list
    const list = data.top_matches
      .map(
        (item) => `
          <li>
            <div class="match-info">
              <h4>${item.event_name}</h4>
              <p>${item.label} • Keywords: ${item.keywords.slice(0, 4).join(", ")}</p>
            </div>
            <div class="match-sim">${Math.round(item.similarity * 100)}%</div>
          </li>
        `
      )
      .join("");

    // Build the Main Result (Classification focus)
    const resultHtml = `
      <div class="classification-box">
        <div class="badge">${prediction.label}</div>
        <div class="event-details">
          <h3 class="event-name">${prediction.event_name}</h3>
          <div class="meta-grid">
            <div class="meta-item">
              <div class="meta-label">Confidence</div>
              <div class="meta-value">${prediction.confidence.toUpperCase()} (${Math.round(prediction.similarity * 100)}%)</div>
            </div>
            <div class="meta-item">
              <div class="meta-label">News Coverage</div>
              <div class="meta-value">${prediction.news_count} Articles</div>
            </div>
          </div>
        </div>
      </div>
      
      <div style="margin-top: 40px;">
        <label>Similar Detected Topics</label>
        <ul class="match-list">${list}</ul>
      </div>
    `;

    showResult(detectResult, resultHtml);
    
    // Smooth scroll to result
    detectResult.scrollIntoView({ behavior: 'smooth' });

  } catch (error) {
    showResult(detectResult, `<p class='status-message error'>ERROR: ${error.message.toUpperCase()}</p>`);
  }
});

// Handle Example Cards
document.querySelectorAll(".example-card").forEach(card => {
  card.addEventListener("click", () => {
    const tweet = card.getAttribute("data-tweet");
    tweetInput.value = tweet;
    // Trigger detection
    document.getElementById("detect-button").click();
  });
});

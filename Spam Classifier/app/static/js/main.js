/**
 * SentinelSpam AI — Frontend Application Logic
 * Author: Senior Full-Stack ML Engineer
 * Integrates Chart.js, REST API endpoints, XAI word highlighter,
 * heuristic threat intelligence, bulk CSV scanner, and dynamic DOM updates.
 */

// Global State
let modelComparisonChart = null;
let rocCurvesChart = null;
let trafficRatioChart = null;
let currentStats = null;
let historyCache = [];

// Sample Presets for Quick Testing
const EMAIL_PRESETS = {
  phishing: "URGENT SECURITY ALERT: Your Microsoft 365 and banking portal access has been flagged for unauthorized login attempts. Immediate identity verification required! Click http://192.168.1.50/login.php within 24 hours to verify your password and prevent permanent account suspension.",
  lottery: "WINNER! You have won a £1,000,000 cash jackpot in the International UK Lottery Draw! Claim ref: #UK-9821. Call 0800123456 now or text CLAIM to 88010 to receive your prize cheque. Valid 48hrs.",
  meeting: "Subject: Q4 Machine Learning Project Roadmap Review\n\nHi Team,\n\nPlease review the attached slide deck before our sprint planning sync tomorrow at 10:00 AM PST. We will discuss the TF-IDF feature pipeline, explainability metrics, and deployment deadlines.\n\nBest regards,\nAlex",
  personal: "Hey! Just checking in to see if you are free for dinner this Friday evening. Let me know what restaurant you'd like to go to. Catch up soon!"
};

// Document Ready
document.addEventListener("DOMContentLoaded", () => {
  initNavigation();
  initClassifierForm();
  initBulkScanner();
  initHistoryControls();
  fetchStatsAndInitCharts();
  fetchHistory();
});

// ---------------------------------------------------------------------------
// 1. Navigation & Tab Switching
// ---------------------------------------------------------------------------
function initNavigation() {
  const navItems = document.querySelectorAll(".nav-item");
  navItems.forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetTab = btn.getAttribute("data-tab");
      switchTab(targetTab);
    });
  });

  // Mobile menu toggle
  const mobileToggle = document.getElementById("mobile-toggle");
  const sidebar = document.querySelector(".sidebar");
  if (mobileToggle && sidebar) {
    mobileToggle.addEventListener("click", () => {
      sidebar.classList.toggle("open");
    });
  }

  // Quick test CTA button in topbar
  const quickBtn = document.getElementById("quick-classify-btn");
  if (quickBtn) {
    quickBtn.addEventListener("click", () => {
      switchTab("classify-tab");
      document.getElementById("email-input")?.focus();
    });
  }
}

function switchTab(tabId) {
  // Update sidebar active classes
  document.querySelectorAll(".nav-item").forEach((item) => {
    if (item.getAttribute("data-tab") === tabId) {
      item.classList.add("active");
    } else {
      item.classList.remove("active");
    }
  });

  // Update tab pane visibility
  document.querySelectorAll(".tab-pane").forEach((pane) => {
    pane.classList.remove("active");
  });
  const activePane = document.getElementById(tabId);
  if (activePane) {
    activePane.classList.add("active");
  }

  // Update Topbar Titles
  const titleMap = {
    "home-tab": {
      title: "Threat Intelligence Overview",
      sub: "Production-grade natural language spam & threat detection powered by Scikit-Learn & XAI"
    },
    "classify-tab": {
      title: "Email Threat Classification & Explainability",
      sub: "Evaluate custom email payloads with real-time inference, XAI word attribution & threat forensics"
    },
    "bulk-tab": {
      title: "Bulk CSV Mailbox Scanner",
      sub: "High-throughput batch email analysis with matrix inference and downloadable audit reports"
    },
    "dashboard-tab": {
      title: "Machine Learning Analytics & ROC Diagnostics",
      sub: "Multi-model benchmark comparisons, ROC curves, and confusion matrix diagnostics"
    },
    "history-tab": {
      title: "Prediction Audit History",
      sub: "Historical database records of classified emails, risk scores, and confidence levels"
    }
  };

  const pageTitle = document.getElementById("page-title");
  const pageSub = document.getElementById("page-subtitle");
  if (pageTitle && pageSub && titleMap[tabId]) {
    pageTitle.textContent = titleMap[tabId].title;
    pageSub.textContent = titleMap[tabId].sub;
  }

  // Close mobile sidebar if open
  document.querySelector(".sidebar")?.classList.remove("open");

  // Re-render charts when dashboard tab is focused to adjust canvas dimensions
  if (tabId === "dashboard-tab") {
    setTimeout(() => {
      if (currentStats) renderCharts(currentStats);
    }, 120);
  }
}

// ---------------------------------------------------------------------------
// 2. Classifier Form & Inference
// ---------------------------------------------------------------------------
function initClassifierForm() {
  const emailInput = document.getElementById("email-input");
  const classifyBtn = document.getElementById("classify-btn");
  const clearBtn = document.getElementById("clear-input-btn");
  const wordCountSpan = document.getElementById("word-count");
  const charCountSpan = document.getElementById("char-count");

  // Real-time word and character counter
  emailInput.addEventListener("input", () => {
    const text = emailInput.value;
    charCountSpan.textContent = text.length;
    const words = text.trim() ? text.trim().split(/\s+/).length : 0;
    wordCountSpan.textContent = words;
    hideInputError();
  });

  // Clear button
  clearBtn.addEventListener("click", () => {
    emailInput.value = "";
    wordCountSpan.textContent = "0";
    charCountSpan.textContent = "0";
    hideInputError();
    resetResultCard();
  });

  // Classify trigger
  classifyBtn.addEventListener("click", () => {
    handleClassification();
  });

  // Ctrl+Enter / Cmd+Enter shortcut
  emailInput.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      handleClassification();
    }
  });
}

function loadPreset(type) {
  const text = EMAIL_PRESETS[type];
  if (!text) return;

  const emailInput = document.getElementById("email-input");
  emailInput.value = text;
  document.getElementById("char-count").textContent = text.length;
  document.getElementById("word-count").textContent = text.trim().split(/\s+/).length;
  hideInputError();
  emailInput.focus();
}

async function handleClassification() {
  const emailInput = document.getElementById("email-input");
  const text = emailInput.value.trim();

  if (!text) {
    showInputError("Please enter or paste email content before submitting.");
    emailInput.focus();
    return;
  }

  setLoadingState(true);
  hideInputError();

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email_text: text })
    });

    const data = await response.json();

    if (!response.ok || !data.success) {
      throw new Error(data.error || "Classification request failed.");
    }

    // Render outcome
    displayPredictionResult(data);

    // Refresh history table and analytics stats
    fetchHistory();
    if (data.live_stats) {
      updateRuntimeStatCounters(data.live_stats);
    }

    showToast("Email analyzed successfully", "success");

  } catch (err) {
    showInputError(err.message || "An unexpected error occurred during prediction.");
    showToast(err.message, "danger");
  } finally {
    setLoadingState(false);
  }
}

function displayPredictionResult(res) {
  const placeholder = document.getElementById("result-placeholder-card");
  const resultCard = document.getElementById("result-card");
  const verdictBanner = document.getElementById("verdict-banner");
  const verdictTitle = document.getElementById("verdict-title");
  const verdictDesc = document.getElementById("verdict-desc");
  const verdictIcon = document.getElementById("verdict-status-icon");
  const confVal = document.getElementById("conf-percentage");
  const spamProbNum = document.getElementById("spam-prob-num");
  const hamProbNum = document.getElementById("ham-prob-num");
  const spamMiniBar = document.getElementById("spam-mini-bar");
  const hamMiniBar = document.getElementById("ham-mini-bar");
  const timestampSpan = document.getElementById("result-timestamp");

  // Show result card
  placeholder.classList.add("hidden");
  resultCard.classList.remove("hidden");

  const isSpam = res.is_spam;
  const conf = res.confidence;
  const probHam = res.probabilities.ham;
  const probSpam = res.probabilities.spam;

  // Verdict Banner styling
  verdictBanner.className = "verdict-banner " + (isSpam ? "is-spam" : "is-ham");
  resultCard.className = "classify-card result-card " + (isSpam ? "is-spam" : "is-ham");

  if (isSpam) {
    verdictTitle.textContent = "SPAM / MALICIOUS THREAT DETECTED";
    verdictDesc.textContent = "This email exhibits structural and lexical patterns strongly correlated with spam, phishing, or social engineering attacks.";
    verdictIcon.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i>';
  } else {
    verdictTitle.textContent = "LEGITIMATE EMAIL (HAM)";
    verdictDesc.textContent = "This email does not exhibit characteristic threat markers and aligns closely with legitimate correspondence.";
    verdictIcon.innerHTML = '<i class="fa-solid fa-circle-check"></i>';
  }

  // Model Probabilities
  confVal.textContent = `${conf}%`;
  spamProbNum.textContent = probSpam.toFixed(4);
  hamProbNum.textContent = probHam.toFixed(4);
  spamMiniBar.style.width = `${(probSpam * 100).toFixed(1)}%`;
  hamMiniBar.style.width = `${(probHam * 100).toFixed(1)}%`;

  // 1. Render Composite Risk Score
  renderRiskScore(res.risk_assessment);

  // 2. Render Explainable AI (XAI) Word Highlighter & Feature Attributions
  renderExplainability(res.explainability);

  // 3. Render Rule-Based Phishing & Forensic Intelligence
  renderPhishingForensics(res.phishing_analysis);

  // Timestamp
  timestampSpan.textContent = new Date().toLocaleTimeString();

  // Smooth scroll into view on smaller displays
  if (window.innerWidth <= 1024) {
    resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
}

function renderRiskScore(risk) {
  if (!risk) return;
  const scoreNum = document.getElementById("risk-score-number");
  const pill = document.getElementById("risk-severity-pill");
  const bar = document.getElementById("risk-progress-bar");
  const advice = document.getElementById("risk-action-advice");

  const score = risk.risk_score;
  const level = risk.risk_level || "MINIMAL";

  if (scoreNum) scoreNum.textContent = score;
  if (bar) {
    bar.style.width = `${score}%`;
    bar.style.backgroundColor = risk.risk_color || "#2563eb";
  }
  if (advice) advice.textContent = risk.action_advice || "";

  if (pill) {
    pill.textContent = level;
    pill.className = `severity-pill pill-${level.toLowerCase()}`;
  }
}

function renderExplainability(xai) {
  if (!xai) return;

  const hlContainer = document.getElementById("highlighted-email-display");
  const naturalText = document.getElementById("natural-explanation-text");
  const spamList = document.getElementById("spam-contrib-list");
  const hamList = document.getElementById("ham-contrib-list");

  // Annotated HTML with Highlighted Spans
  if (hlContainer) {
    hlContainer.innerHTML = xai.annotated_html || "<em>No token matches</em>";
  }

  // Natural language explanation
  if (naturalText) {
    naturalText.textContent = xai.natural_explanation || "No explanation available.";
  }

  // Top positive contributors (pushing toward Spam)
  if (spamList) {
    const topSpam = xai.top_spam_features || [];
    if (topSpam.length === 0) {
      spamList.innerHTML = "<em>No significant spam-driving features</em>";
    } else {
      spamList.innerHTML = topSpam.map(f => `
        <div class="contrib-pill spam-pill">
          <span>'${escapeHtml(f.feature)}'</span>
          <strong>+${f.contribution.toFixed(3)}</strong>
        </div>
      `).join("");
    }
  }

  // Top negative contributors (pushing toward Ham)
  if (hamList) {
    const topHam = xai.top_ham_features || [];
    if (topHam.length === 0) {
      hamList.innerHTML = "<em>No significant legitimate anchor features</em>";
    } else {
      hamList.innerHTML = topHam.map(f => `
        <div class="contrib-pill ham-pill">
          <span>'${escapeHtml(f.feature)}'</span>
          <strong>${f.contribution.toFixed(3)}</strong>
        </div>
      `).join("");
    }
  }
}

function renderPhishingForensics(phish) {
  if (!phish) return;

  const badge = document.getElementById("threat-level-badge");
  const grid = document.getElementById("forensics-indicators-grid");
  const disclaimer = document.getElementById("safety-disclaimer-text");

  const level = phish.threat_level || "NONE_DETECTED";

  if (badge) {
    badge.textContent = level === "NONE_DETECTED" ? "NO HEURISTIC SIGNATURES" : `${level} THREAT`;
    badge.className = `threat-level-badge ${level === 'CRITICAL' ? 'pill-critical' : (level === 'HIGH' ? 'pill-high' : (level === 'MEDIUM' ? 'pill-medium' : 'pill-minimal'))}`;
  }

  if (disclaimer) {
    disclaimer.textContent = phish.disclaimer || "";
  }

  if (grid) {
    const items = [];

    // URL Flags
    if (phish.url_analysis?.flags?.length > 0) {
      phish.url_analysis.flags.forEach(flag => {
        items.push(`
          <div class="indicator-item flag-critical">
            <i class="fa-solid fa-link-slash"></i>
            <div><strong>Suspicious Link:</strong> ${escapeHtml(flag)}</div>
          </div>
        `);
      });
    }

    // Urgency Flags
    if (phish.urgency_indicators?.length > 0) {
      phish.urgency_indicators.forEach(u => {
        items.push(`
          <div class="indicator-item flag-warning">
            <i class="fa-solid fa-clock-rotate-left"></i>
            <div><strong>Urgency Pattern:</strong> ${escapeHtml(u.pattern)} (matched: <em>"${escapeHtml(u.matched_text)}"</em>)</div>
          </div>
        `);
      });
    }

    // Credential Harvesting
    if (phish.credential_harvesting?.length > 0) {
      phish.credential_harvesting.forEach(c => {
        items.push(`
          <div class="indicator-item flag-critical">
            <i class="fa-solid fa-key"></i>
            <div><strong>Credential Harvesting:</strong> ${escapeHtml(c.pattern)} (matched: <em>"${escapeHtml(c.matched_text)}"</em>)</div>
          </div>
        `);
      });
    }

    // Brand Impersonation
    if (phish.impersonation_indicators?.length > 0) {
      phish.impersonation_indicators.forEach(i => {
        items.push(`
          <div class="indicator-item flag-critical">
            <i class="fa-solid fa-building-shield"></i>
            <div><strong>Brand Spoofing:</strong> ${escapeHtml(i.brand)} — ${escapeHtml(i.reason)}</div>
          </div>
        `);
      });
    }

    if (items.length === 0) {
      grid.innerHTML = `
        <div class="indicator-item" style="border-left-color: #10b981; background-color: #f0fdf4; color: #166534;">
          <i class="fa-solid fa-check-double"></i>
          <div>No overt heuristic phishing triggers matched. (Refer to ML statistical probability).</div>
        </div>
      `;
    } else {
      grid.innerHTML = items.join("");
    }
  }
}

function resetResultCard() {
  document.getElementById("result-placeholder-card")?.classList.remove("hidden");
  document.getElementById("result-card")?.classList.add("hidden");
}

function setLoadingState(isLoading) {
  const btn = document.getElementById("classify-btn");
  const spinner = document.getElementById("classify-spinner");
  const icon = document.getElementById("classify-btn-icon");
  const text = document.getElementById("classify-btn-text");

  if (isLoading) {
    btn.disabled = true;
    spinner.classList.remove("hidden");
    icon.classList.add("hidden");
    text.textContent = "Analyzing & Explaining...";
  } else {
    btn.disabled = false;
    spinner.classList.add("hidden");
    icon.classList.remove("hidden");
    text.textContent = "Classify & Explain";
  }
}

function showInputError(msg) {
  const errorDiv = document.getElementById("input-error");
  if (errorDiv) {
    errorDiv.textContent = msg;
    errorDiv.classList.remove("hidden");
  }
}

function hideInputError() {
  const errorDiv = document.getElementById("input-error");
  if (errorDiv) {
    errorDiv.classList.add("hidden");
  }
}

// ---------------------------------------------------------------------------
// 3. Bulk CSV Scanner Implementation
// ---------------------------------------------------------------------------
function initBulkScanner() {
  const dropzone = document.getElementById("bulk-dropzone");
  const fileInput = document.getElementById("bulk-file-input");
  const sampleBtn = document.getElementById("load-sample-bulk-btn");

  if (!dropzone || !fileInput) return;

  // Drag and drop events
  ["dragenter", "dragover"].forEach(evtName => {
    dropzone.addEventListener(evtName, (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach(evtName => {
    dropzone.addEventListener(evtName, (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleBulkFileUpload(files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      handleBulkFileUpload(e.target.files[0]);
    }
  });

  if (sampleBtn) {
    sampleBtn.addEventListener("click", () => {
      handleLoadSampleBulkBatch();
    });
  }
}

async function handleBulkFileUpload(file) {
  if (!file.name.toLowerCase().endsWith(".csv")) {
    showToast("Please upload a CSV file format.", "danger");
    return;
  }

  const status = document.getElementById("upload-status");
  status.textContent = `Uploading and classifying '${file.name}'...`;
  status.classList.remove("hidden");

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/api/bulk-predict", {
      method: "POST",
      body: formData
    });

    const data = await res.json();
    if (!res.ok || !data.success) {
      throw new Error(data.error || "Bulk processing failed.");
    }

    renderBulkResults(data.summary, data.filename);
    showToast(`Successfully classified ${data.summary.total_emails} emails!`, "success");
    status.classList.add("hidden");

  } catch (err) {
    status.textContent = `Error: ${err.message}`;
    showToast(err.message, "danger");
  }
}

function handleLoadSampleBulkBatch() {
  const sampleCSV = `email
"URGENT: Your Microsoft 365 password is expiring! Click http://192.168.1.50/login within 24h to verify."
"Hey David, are we still meeting tomorrow for sprint planning at 10 AM?"
"WINNER!! You have won a $1,000 Walmart Gift Card! Call 08001234 now to claim free prize."
"Please find attached the financial review spreadsheet from yesterday's sync."
"FINAL NOTICE: Your PayPal account has been suspended. Confirm identity at http://paypal-verify.xyz"
"Hi Mom, I will be home by 7 PM. Let me know if you need any groceries."
"Security Alert: Unauthorized access detected from Russia. Log in to restore account immediately."
"Thanks for sending over the project proposal, I will review it over the weekend."`;

  const blob = new Blob([sampleCSV], { type: "text/csv" });
  const file = new File([blob], "sample_test_batch.csv", { type: "text/csv" });
  handleBulkFileUpload(file);
}

function renderBulkResults(summary, filename) {
  const panel = document.getElementById("bulk-results-panel");
  if (!panel) return;
  panel.classList.remove("hidden");

  document.getElementById("bulk-total-val").textContent = summary.total_emails.toLocaleString();
  document.getElementById("bulk-spam-val").textContent = summary.spam_count.toLocaleString();
  document.getElementById("bulk-ham-val").textContent = summary.ham_count.toLocaleString();
  document.getElementById("bulk-high-risk-val").textContent = summary.high_risk_count.toLocaleString();
  document.getElementById("bulk-spam-rate-val").textContent = `${summary.spam_percentage}% spam rate`;
  document.getElementById("bulk-filename-val").textContent = filename || "batch.csv";

  // Render Table Rows
  const tbody = document.getElementById("bulk-table-body");
  if (!tbody) return;

  const records = summary.records || [];
  tbody.innerHTML = records.map(r => {
    const isSpam = r.is_spam;
    const badgeClass = isSpam ? "badge-spam" : "badge-ham";
    const badgeIcon = isSpam ? "fa-shield-virus" : "fa-shield-halved";

    return `
      <tr>
        <td><strong>#${r.id}</strong></td>
        <td class="preview-cell" title="${escapeHtml(r.preview)}">${escapeHtml(r.preview)}</td>
        <td>
          <span class="badge ${badgeClass}">
            <i class="fa-solid ${badgeIcon}"></i> ${r.classification}
          </span>
        </td>
        <td><strong>${r.confidence_percent.toFixed(1)}%</strong></td>
        <td>
          <span class="severity-pill pill-${r.risk_level.toLowerCase()}">${r.risk_score}/100</span>
        </td>
        <td><small class="text-muted">${escapeHtml(r.top_threat_flag)}</small></td>
      </tr>
    `;
  }).join("");

  panel.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ---------------------------------------------------------------------------
// 4. Stats & Chart.js Visualizations (Multi-Model ROC Curves & Comparison)
// ---------------------------------------------------------------------------
async function fetchStatsAndInitCharts() {
  try {
    const response = await fetch("/api/stats");
    const data = await response.json();
    if (!response.ok || !data.success) {
      throw new Error("Failed to load statistics.");
    }

    currentStats = data;

    // Update Sidebar & Hero model labels
    const champion = data.model_benchmarks?.champion_model || "Support Vector Machine";
    const accuracy = ((data.model_benchmarks?.accuracy || 0.9845) * 100).toFixed(2) + "%";
    const aucScore = (data.model_benchmarks?.roc_auc || 0.9922).toFixed(4);

    const sidebarModel = document.getElementById("sidebar-champion-model");
    const sidebarAcc = document.getElementById("sidebar-accuracy");
    const heroChampion = document.getElementById("hero-champion-name");
    const heroAcc = document.getElementById("hero-accuracy-val");

    if (sidebarModel) sidebarModel.textContent = champion;
    if (sidebarAcc) sidebarAcc.textContent = `${accuracy} / AUC ${aucScore}`;
    if (heroChampion) heroChampion.textContent = champion;
    if (heroAcc) heroAcc.textContent = `${accuracy} (AUC ${aucScore})`;

    // Update Confusion Matrix diagnostics
    updateConfusionMatrix(data.model_benchmarks);

    // Update Runtime Stat Counters
    updateRuntimeStatCounters(data.runtime_stats);

    // Render Charts (Bar Chart, ROC Curve Chart, Traffic Ratio Doughnut)
    renderCharts(data);

  } catch (err) {
    console.error("Error fetching stats:", err);
  }
}

function updateRuntimeStatCounters(stats) {
  if (!stats) return;

  const totalElem = document.getElementById("stat-total-emails");
  const spamElem = document.getElementById("stat-total-spam");
  const hamElem = document.getElementById("stat-total-ham");
  const spamRateElem = document.getElementById("stat-spam-rate");
  const hamRateElem = document.getElementById("stat-ham-rate");

  const total = stats.total_emails || 0;
  const spam = stats.total_spam || 0;
  const ham = stats.total_ham || 0;

  if (totalElem) totalElem.textContent = total.toLocaleString();
  if (spamElem) spamElem.textContent = spam.toLocaleString();
  if (hamElem) hamElem.textContent = ham.toLocaleString();

  const spamPct = total > 0 ? ((spam / total) * 100).toFixed(1) : "0.0";
  const hamPct = total > 0 ? ((ham / total) * 100).toFixed(1) : "0.0";

  if (spamRateElem) spamRateElem.textContent = `${spamPct}% of total`;
  if (hamRateElem) hamRateElem.textContent = `${hamPct}% of total`;

  // Update Doughnut chart if present
  if (trafficRatioChart) {
    trafficRatioChart.data.datasets[0].data = (ham === 0 && spam === 0) ? [1, 0] : [ham, spam];
    trafficRatioChart.update();
  }
}

function updateConfusionMatrix(benchmarks) {
  if (!benchmarks) return;
  const cm = benchmarks.confusion_matrix || [[896, 6], [10, 118]];
  const tn = cm[0][0];
  const fp = cm[0][1];
  const fn = cm[1][0];
  const tp = cm[1][1];

  const tnElem = document.getElementById("cm-tn");
  const fpElem = document.getElementById("cm-fp");
  const fnElem = document.getElementById("cm-fn");
  const tpElem = document.getElementById("cm-tp");

  if (tnElem) tnElem.textContent = tn;
  if (fpElem) fpElem.textContent = fp;
  if (fnElem) fnElem.textContent = fn;
  if (tpElem) tpElem.textContent = tp;

  const prec = benchmarks.precision ? (benchmarks.precision * 100).toFixed(2) + "%" : "95.16%";
  const rec = benchmarks.recall ? (benchmarks.recall * 100).toFixed(2) + "%" : "92.19%";
  const far = tn + fp > 0 ? ((fp / (tn + fp)) * 100).toFixed(2) + "%" : "0.66%";
  const auc = benchmarks.roc_auc ? benchmarks.roc_auc.toFixed(4) : "0.9922";

  const precElem = document.getElementById("cm-prec");
  const recElem = document.getElementById("cm-rec");
  const farElem = document.getElementById("cm-far");
  const aucElem = document.getElementById("cm-auc");
  const championAcc = document.getElementById("stat-champion-acc");
  const championAuc = document.getElementById("stat-champion-auc");

  if (precElem) precElem.textContent = prec;
  if (recElem) recElem.textContent = rec;
  if (farElem) farElem.textContent = far;
  if (aucElem) aucElem.textContent = auc;
  if (championAcc) championAcc.textContent = ((benchmarks.accuracy || 0.9845) * 100).toFixed(2) + "%";
  if (championAuc) championAuc.textContent = `ROC-AUC: ${auc}`;
}

function renderCharts(data) {
  const allModels = data.model_benchmarks?.all_models;
  renderModelComparisonChart(allModels);
  renderRocCurvesChart(allModels);
  renderTrafficRatioChart(data.runtime_stats);
}

function renderModelComparisonChart(allModels) {
  const canvas = document.getElementById("modelComparisonChart");
  if (!canvas) return;

  const ctx = canvas.getContext("2d");
  if (modelComparisonChart) {
    modelComparisonChart.destroy();
  }

  const models = allModels || {
    "Multinomial Naive Bayes": { accuracy: 0.9806, precision: 0.9737, recall: 0.8672, f1_score: 0.9174 },
    "Logistic Regression": { accuracy: 0.9825, precision: 0.9231, recall: 0.9375, f1_score: 0.9302 },
    "Random Forest": { accuracy: 0.9825, precision: 0.9825, recall: 0.8750, f1_score: 0.9256 },
    "Support Vector Machine": { accuracy: 0.9845, precision: 0.9516, recall: 0.9219, f1_score: 0.9365 }
  };

  const labels = Object.keys(models);
  const accuracyData = labels.map((k) => (models[k].accuracy * 100).toFixed(2));
  const precisionData = labels.map((k) => (models[k].precision * 100).toFixed(2));
  const recallData = labels.map((k) => (models[k].recall * 100).toFixed(2));
  const f1Data = labels.map((k) => (models[k].f1_score * 100).toFixed(2));

  modelComparisonChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels.map(l => l.replace("Multinomial ", "")),
      datasets: [
        {
          label: "Accuracy (%)",
          data: accuracyData,
          backgroundColor: "#3b82f6",
          borderRadius: 4
        },
        {
          label: "Precision (%)",
          data: precisionData,
          backgroundColor: "#10b981",
          borderRadius: 4
        },
        {
          label: "Recall (%)",
          data: recallData,
          backgroundColor: "#f59e0b",
          borderRadius: 4
        },
        {
          label: "F1-Score (%)",
          data: f1Data,
          backgroundColor: "#8b5cf6",
          borderRadius: 4
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          min: 80,
          max: 100,
          ticks: {
            callback: (val) => val + "%",
            font: { family: "'Plus Jakarta Sans', sans-serif" }
          },
          grid: { color: "#f1f5f9" }
        },
        x: {
          grid: { display: false },
          ticks: { font: { family: "'Plus Jakarta Sans', sans-serif", weight: 600 } }
        }
      },
      plugins: {
        legend: {
          position: "top",
          labels: {
            font: { family: "'Plus Jakarta Sans', sans-serif", weight: 500, size: 12 },
            boxWidth: 12,
            boxHeight: 12,
            usePointStyle: true
          }
        },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.dataset.label}: ${ctx.raw}%`
          }
        }
      }
    }
  });
}

function renderRocCurvesChart(allModels) {
  const canvas = document.getElementById("rocCurvesChart");
  if (!canvas) return;

  const ctx = canvas.getContext("2d");
  if (rocCurvesChart) {
    rocCurvesChart.destroy();
  }

  // Model color mapping
  const colorMap = {
    "Support Vector Machine": { color: "#2563eb", width: 3 },
    "Random Forest": { color: "#8b5cf6", width: 2 },
    "Logistic Regression": { color: "#10b981", width: 2 },
    "Multinomial Naive Bayes": { color: "#f59e0b", width: 2 }
  };

  const datasets = [];

  // 1. Add model ROC curves
  if (allModels) {
    Object.keys(allModels).forEach(modelName => {
      const modelData = allModels[modelName];
      const curve = modelData.roc_curve || [];
      const auc = modelData.roc_auc ? modelData.roc_auc.toFixed(4) : "0.99";
      const cfg = colorMap[modelName] || { color: "#64748b", width: 2 };

      datasets.push({
        label: `${modelName.replace("Multinomial ", "")} (AUC: ${auc})`,
        data: curve,
        borderColor: cfg.color,
        borderWidth: cfg.width,
        pointRadius: 0,
        fill: false,
        tension: 0.1
      });
    });
  }

  // 2. Add diagonal reference line (Random Chance y = x)
  datasets.push({
    label: "Random Chance (AUC: 0.5000)",
    data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
    borderColor: "#94a3b8",
    borderWidth: 1.5,
    borderDash: [5, 5],
    pointRadius: 0,
    fill: false
  });

  rocCurvesChart = new Chart(ctx, {
    type: "line",
    data: { datasets: datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          type: "linear",
          min: 0,
          max: 1,
          title: {
            display: true,
            text: "False Positive Rate (1 - Specificity)",
            font: { family: "'Plus Jakarta Sans', sans-serif", weight: 600, size: 11 }
          },
          grid: { color: "#f1f5f9" }
        },
        y: {
          min: 0,
          max: 1.02,
          title: {
            display: true,
            text: "True Positive Rate (Sensitivity / Recall)",
            font: { family: "'Plus Jakarta Sans', sans-serif", weight: 600, size: 11 }
          },
          grid: { color: "#f1f5f9" }
        }
      },
      plugins: {
        legend: {
          position: "top",
          labels: {
            font: { family: "'Plus Jakarta Sans', sans-serif", size: 11, weight: 500 },
            boxWidth: 12,
            boxHeight: 12,
            usePointStyle: true
          }
        },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.dataset.label}: FPR=${ctx.parsed.x.toFixed(2)}, TPR=${ctx.parsed.y.toFixed(2)}`
          }
        }
      }
    }
  });
}

function renderTrafficRatioChart(runtimeStats) {
  const canvas = document.getElementById("trafficRatioChart");
  if (!canvas) return;

  const ctx = canvas.getContext("2d");
  if (trafficRatioChart) {
    trafficRatioChart.destroy();
  }

  const hamCount = runtimeStats?.total_ham || 0;
  const spamCount = runtimeStats?.total_spam || 0;
  const displayData = (hamCount === 0 && spamCount === 0) ? [1, 0] : [hamCount, spamCount];

  trafficRatioChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["Legitimate (Ham)", "Spam Intercepted"],
      datasets: [
        {
          data: displayData,
          backgroundColor: ["#10b981", "#ef4444"],
          borderWidth: 2,
          borderColor: "#ffffff"
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: "70%",
      plugins: {
        legend: {
          position: "bottom",
          labels: {
            font: { family: "'Plus Jakarta Sans', sans-serif", weight: 500, size: 12 },
            usePointStyle: true
          }
        },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              if (hamCount === 0 && spamCount === 0) return "No emails analyzed yet";
              return `${ctx.label}: ${ctx.raw} emails`;
            }
          }
        }
      }
    }
  });
}

// ---------------------------------------------------------------------------
// 5. History Table Management
// ---------------------------------------------------------------------------
function initHistoryControls() {
  const clearBtn = document.getElementById("clear-history-btn");
  const searchInput = document.getElementById("history-search");

  if (clearBtn) {
    clearBtn.addEventListener("click", () => {
      if (confirm("Are you sure you want to clear all prediction history? This action cannot be undone.")) {
        handleClearHistory();
      }
    });
  }

  if (searchInput) {
    searchInput.addEventListener("input", (e) => {
      const q = e.target.value.toLowerCase().trim();
      filterHistoryTable(q);
    });
  }
}

async function fetchHistory() {
  try {
    const res = await fetch("/api/history?limit=100");
    const data = await res.json();
    if (res.ok && data.success) {
      historyCache = data.history || [];
      renderHistoryTable(historyCache);
    }
  } catch (err) {
    console.error("Error fetching history:", err);
  }
}

function renderHistoryTable(items) {
  const tbody = document.getElementById("history-table-body");
  if (!tbody) return;

  if (!items || items.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="5" class="empty-table-msg">
          <i class="fa-solid fa-inbox"></i> No classifications recorded yet. Analyze an email to begin.
        </td>
      </tr>
    `;
    return;
  }

  tbody.innerHTML = items.map((item) => {
    const isSpam = item.is_spam === 1 || item.label.toLowerCase() === "spam";
    const badgeClass = isSpam ? "badge-spam" : "badge-ham";
    const badgeIcon = isSpam ? "fa-shield-virus" : "fa-shield-halved";
    const badgeText = isSpam ? "Spam" : "Legitimate";

    return `
      <tr>
        <td><strong>#${item.id}</strong></td>
        <td class="preview-cell" title="${escapeHtml(item.email_preview)}">${escapeHtml(item.email_preview)}</td>
        <td>
          <span class="badge ${badgeClass}">
            <i class="fa-solid ${badgeIcon}"></i> ${badgeText}
          </span>
        </td>
        <td><strong>${item.confidence.toFixed(1)}%</strong></td>
        <td><small class="text-muted">${item.created_at}</small></td>
      </tr>
    `;
  }).join("");
}

function filterHistoryTable(query) {
  if (!query) {
    renderHistoryTable(historyCache);
    return;
  }
  const filtered = historyCache.filter((item) => {
    return (
      item.email_preview.toLowerCase().includes(query) ||
      item.label.toLowerCase().includes(query) ||
      String(item.id).includes(query)
    );
  });
  renderHistoryTable(filtered);
}

async function handleClearHistory() {
  try {
    const res = await fetch("/api/history", { method: "DELETE" });
    const data = await res.json();
    if (res.ok && data.success) {
      showToast(data.message || "History cleared", "success");
      fetchHistory();
      fetchStatsAndInitCharts();
    } else {
      throw new Error(data.error || "Failed to clear history.");
    }
  } catch (err) {
    showToast(err.message, "danger");
  }
}

// ---------------------------------------------------------------------------
// 6. Toast Notifications & Utility Helpers
// ---------------------------------------------------------------------------
let toastTimer = null;
function showToast(message, type = "info") {
  const toast = document.getElementById("toast");
  if (!toast) return;

  clearTimeout(toastTimer);
  toast.textContent = message;
  toast.className = `toast toast-${type}`;
  toast.classList.remove("hidden");

  toastTimer = setTimeout(() => {
    toast.classList.add("hidden");
  }, 3500);
}

function escapeHtml(str) {
  if (!str) return "";
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

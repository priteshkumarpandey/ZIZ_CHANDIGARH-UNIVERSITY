import fs from "fs";
import path from "path";
import { spawn } from "child_process";
import { createRequire } from "module";

const require = createRequire(import.meta.url);
const { chromium } = require("../frontend/node_modules/playwright");

const ROOT_DIR = path.resolve(".");
const SCREENSHOTS_DIR = path.join(ROOT_DIR, "docs", "screenshots");
const THRESHOLDS_FILE = path.join(ROOT_DIR, "reports", "operating_thresholds.json");

if (!fs.existsSync(SCREENSHOTS_DIR)) {
  fs.mkdirSync(SCREENSHOTS_DIR, { recursive: true });
}

// 240s global timeout protection
const globalTimeout = setTimeout(() => {
  console.error("Global 240s timeout reached in capture_screenshots.mjs. Terminating.");
  process.exit(1);
}, 240000);
globalTimeout.unref();

async function isServerRunning(url) {
  try {
    const res = await fetch(url);
    return res.status === 200 || res.status === 404;
  } catch {
    return false;
  }
}

async function ensureBackendRunning() {
  const isHealthy = await isServerRunning("http://localhost:8000/health");
  if (isHealthy) {
    console.log("Backend API is running on http://localhost:8000");
    return;
  }

  console.log("Starting Docker Compose stack for backend...");
  const child = spawn("docker", ["compose", "up", "-d", "timescaledb", "kafka", "api"], {
    cwd: ROOT_DIR,
    stdio: "inherit",
    shell: true,
  });

  await new Promise((resolve, reject) => {
    child.on("exit", (code) => {
      if (code === 0) resolve();
      else reject(new Error(`docker compose exited with code ${code}`));
    });
  });

  console.log("Waiting for backend API to become healthy...");
  for (let i = 0; i < 30; i++) {
    await new Promise((r) => setTimeout(r, 1000));
    if (await isServerRunning("http://localhost:8000/health")) {
      console.log("Backend API is healthy on port 8000!");
      return;
    }
  }
  throw new Error("Timeout waiting for backend API on port 8000");
}

async function startNextServer() {
  const isRunning = await isServerRunning("http://localhost:3000");
  if (isRunning) {
    console.log("Next.js server is already running on http://localhost:3000");
    return null;
  }

  console.log("Starting Next.js server on http://localhost:3000...");
  const child = spawn("npx", ["next", "start", "-p", "3000"], {
    cwd: path.join(ROOT_DIR, "frontend"),
    stdio: "pipe",
    shell: true,
  });

  for (let i = 0; i < 30; i++) {
    await new Promise((r) => setTimeout(r, 1000));
    if (await isServerRunning("http://localhost:3000")) {
      console.log("Next.js server is ready on port 3000!");
      return child;
    }
  }
  throw new Error("Timeout waiting for Next.js server to start on port 3000");
}

async function killServer(serverProcess) {
  if (!serverProcess || !serverProcess.pid) return;
  console.log(`Shutting down Next.js server process (PID ${serverProcess.pid})...`);
  try {
    if (process.platform === "win32") {
      spawn("taskkill", ["/pid", serverProcess.pid.toString(), "/f", "/t"]);
    } else {
      serverProcess.kill("SIGTERM");
    }
  } catch (err) {
    console.warn("Server shutdown warning:", err);
  }
}

async function main() {
  let serverProcess = null;
  let browser = null;

  try {
    // 1. Verify / start real backend and frontend
    await ensureBackendRunning();
    serverProcess = await startNextServer();

    console.log("Launching Playwright Chromium...");
    browser = await chromium.launch({
      headless: true,
      args: ["--no-sandbox", "--disable-setuid-sandbox"],
    });

    const context = await browser.newContext({
      viewport: { width: 1440, height: 900 },
      deviceScaleFactor: 1,
    });

    const page = await context.newPage();

    // Passively log all real API responses (zero mocking / zero interception)
    const loggedApiResponses = [];
    page.on("response", async (response) => {
      const url = response.url();
      if (url.includes("/api/v1/")) {
        try {
          const body = await response.json();
          loggedApiResponses.push({ url, status: response.status(), body });
          console.log(`[REAL API RESPONSE] ${response.status()} ${url}`);
        } catch {
          // not json
        }
      }
    });

    // 2. Capture Patients Page from real API
    console.log("Navigating to Patients page (http://localhost:3000/patients)...");
    await page.goto("http://localhost:3000/patients", { waitUntil: "networkidle" });
    await page.waitForSelector("text=Patient Persona Directory", { timeout: 15000 });
    await page.waitForSelector("a:has-text('Launch Digital Twin')", { timeout: 15000 });
    await page.waitForTimeout(500);
    await page.evaluate(() => {
      document.querySelectorAll(".animate-page-enter").forEach((el) => {
        el.style.opacity = "1";
        el.style.transform = "none";
        el.style.animation = "none";
      });
    });

    const patientsPng = path.join(SCREENSHOTS_DIR, "patients.png");
    await page.screenshot({ path: patientsPng, fullPage: false });
    console.log(`Saved Patients page screenshot: ${patientsPng}`);

async function findAlertStep() {
  try {
    const res = await fetch("http://localhost:8000/api/v1/twin/DEMO_PAT_001/cgm?days=3");
    if (!res.ok) {
      console.warn(`Fetch CGM failed with status ${res.status}`);
      return 0;
    }
    const cgm = await res.json();
    if (!Array.isArray(cgm) || cgm.length < 72) {
      console.warn("CGM data returned less than 72 readings");
      return 0;
    }

    console.log(`Searching ${cgm.length - 71} windows for alert onset step...`);
    for (let i = 0; i <= cgm.length - 72; i++) {
      const windowSlice = cgm.slice(i, i + 72);
      const lastReading = windowSlice[windowSlice.length - 1];
      const curr = Number(lastReading.glucose_mgdl ?? lastReading.glucose ?? 120);

      if (curr >= 150 && curr < 180) {
        const series = windowSlice.map((r) => Number(r.glucose_mgdl ?? r.glucose ?? 120));
        const pRes = await fetch("http://localhost:8000/api/v1/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            patient_id: "DEMO_PAT_001",
            recent_cgm_series: series,
            recent_carbs_g: 0,
            recent_glycemic_index: 0,
            recent_heart_rate: 72.0,
            recent_mets: 1.0,
            as_of: lastReading.timestamp,
          }),
        });
        if (pRes.ok) {
          const pData = await pRes.json();
          const horizons = pData.forecast_horizons || [];
          const h30 = horizons[0];
          const isUsable = h30?.alert_usable?.hyperglycemia !== false && Boolean(h30?.alert_usable);
          if (h30 && isUsable && h30.predicted_event && h30.predicted_event !== "none") {
            console.log(`Found alert onset step ${i}: curr_glucose=${curr} mg/dL, event=${h30.predicted_event}, +30m q50=${h30.q50} mg/dL, spike_risk=${h30.spike_risk_probability}`);
            return i;
          }
        }
      }
    }
    console.log("No exact 150-180 window found, defaulting to step 0");
  } catch (err) {
    console.warn("findAlertStep error:", err);
  }
  return 0;
}

    // 3. Capture Twin Page at Alert Onset
    const onsetStep = await findAlertStep();
    console.log(`Navigating to Twin page for DEMO_PAT_001 at onset step ${onsetStep}...`);
    await page.goto(`http://localhost:3000/twin/DEMO_PAT_001?step=${onsetStep}`, { waitUntil: "networkidle" });
    await page.waitForTimeout(1500);

    // Wait for alert card to be rendered by real API prediction
    await page.waitForSelector("text=Hyperglycemia risk in the next 30 min", { timeout: 10000 });

    // Expand Evidence row
    const evidenceButton = (await page.$('button:has-text("Evidence")'))
      || (await page.$('button:has-text("Evidence:")'));
    if (evidenceButton) {
      console.log("Expanding Evidence row on alert card...");
      await evidenceButton.click();
      await page.waitForTimeout(500);
    }

    // 4. Automated check: Verify every metric in the alert card matches reports/operating_thresholds.json
    console.log("Performing automated verification of alert metrics against reports/operating_thresholds.json...");
    const thresholdsData = JSON.parse(fs.readFileSync(THRESHOLDS_FILE, "utf-8"));
    const h30Event = thresholdsData.events.hyperglycemia_30m;
    const h60Event = thresholdsData.events.hyperglycemia_60m;

    if (!h30Event) {
      throw new Error("hyperglycemia_30m missing from reports/operating_thresholds.json");
    }

    // Check alert card text content
    const pageText = await page.textContent("body");

    // Must show exactly the usable +30m alert card
    if (!pageText.includes("Hyperglycemia risk in the next 30 min")) {
      throw new Error("Expected alert card 'Hyperglycemia risk in the next 30 min' not found in UI!");
    }

    // Must NOT show an alert card for +60 min or +120 min
    if (pageText.includes("Hyperglycemia risk in the next 60 min")) {
      throw new Error("Found unexpected alert card for +60 min! +60m is not usable per operating_thresholds.json");
    }
    if (pageText.includes("Hyperglycemia risk in the next 120 min")) {
      throw new Error("Found unexpected alert card for +120 min!");
    }

    // Extract cross-validated mean +- SD metrics from alert card
    const expRecall = `${Math.round(h30Event.recall_mean * 100)} \u00b1 ${Math.round(h30Event.recall_std * 100)}%`;
    const expPrecision = `${Math.round(h30Event.precision_mean * 100)} \u00b1 ${Math.round(h30Event.precision_std * 100)}%`;
    const expFalseAlerts = `${h30Event.false_alerts_per_patient_day_mean.toFixed(1)} \u00b1 ${h30Event.false_alerts_per_patient_day_std.toFixed(1)}/day`;

    const foundRecall = pageText.includes(`Held-out Recall: ${expRecall}`) || pageText.includes(expRecall);
    const foundPrecision = pageText.includes(`Precision: ${expPrecision}`) || pageText.includes(expPrecision);
    const foundFalseAlerts = pageText.includes(`False alerts: ${expFalseAlerts}`) || pageText.includes(expFalseAlerts);
    const foundCrossValNote = pageText.includes("Cross-validated across 5 validation folds (mean \u00b1 SD)")
      || pageText.includes("Cross-validated across 5 validation folds (mean ± SD)");

    if (!foundRecall) {
      throw new Error(`Alert card Recall does not match cross-validated mean \u00b1 SD! Expected '${expRecall}'`);
    }
    if (!foundPrecision) {
      throw new Error(`Alert card Precision does not match cross-validated mean \u00b1 SD! Expected '${expPrecision}'`);
    }
    if (!foundFalseAlerts) {
      throw new Error(`Alert card False alerts does not match cross-validated mean \u00b1 SD! Expected '${expFalseAlerts}'`);
    }
    if (!foundCrossValNote) {
      throw new Error("Alert card cross-validation note 'Cross-validated across 5 validation folds (mean \u00b1 SD)' not found in UI!");
    }

    console.log("PASS: Alert card metrics verified against reports/operating_thresholds.json:");
    console.log(`  - Recall matches cross-validated mean \u00b1 SD: ${expRecall}`);
    console.log(`  - Precision matches cross-validated mean \u00b1 SD: ${expPrecision}`);
    console.log(`  - False alerts/day matches cross-validated mean \u00b1 SD: ${expFalseAlerts}`);
    console.log(`  - Cross-validation note verified: 'Cross-validated across 5 validation folds (mean \u00b1 SD)'`);
    console.log(`  - Unusable +60m event (alert_usable: ${h60Event.alert_usable}) is properly suppressed from actionable alerts`);

    // 4b. Verify Why panel current glucose equals Hero card current glucose
    console.log("Verifying Hero card and Why panel current glucose synchronization...");
    await page.waitForSelector('[data-testid="hero-current-glucose"]', { timeout: 10000 });
    await page.waitForSelector('[data-testid="why-panel-current-glucose"]', { timeout: 10000 });

    const heroGlucoseEl = await page.$('[data-testid="hero-current-glucose"]');
    const whyGlucoseEl = await page.$('[data-testid="why-panel-current-glucose"]');

    if (!heroGlucoseEl || !whyGlucoseEl) {
      throw new Error("Could not find hero-current-glucose or why-panel-current-glucose element!");
    }

    const heroGlucoseText = (await heroGlucoseEl.textContent()).trim();
    const whyGlucoseText = (await whyGlucoseEl.textContent()).trim();
    const heroGlucoseVal = parseFloat(heroGlucoseText);
    const whyGlucoseVal = parseFloat(whyGlucoseText);

    console.log(`Hero card current glucose: ${heroGlucoseVal} mg/dL, Why panel current glucose: ${whyGlucoseVal} mg/dL`);
    if (isNaN(heroGlucoseVal) || isNaN(whyGlucoseVal) || Math.abs(heroGlucoseVal - whyGlucoseVal) > 0.1) {
      throw new Error(
        `Race condition detected! Hero card glucose (${heroGlucoseVal}) does not equal Why panel glucose (${whyGlucoseVal})`
      );
    }
    console.log("PASS: Why panel current glucose strictly equals hero card current glucose.");

    // 5. Capture Twin Top Viewport Screenshot (1440x900 viewport, no scroll)
    const twinTopPng = path.join(SCREENSHOTS_DIR, "twin_top.png");
    await page.screenshot({ path: twinTopPng, fullPage: false });
    console.log(`Saved Twin top viewport screenshot (1440x900): ${twinTopPng}`);

    // Capture Twin Top Viewport Screenshot at 1280x900 viewport
    await page.setViewportSize({ width: 1280, height: 900 });
    await page.waitForTimeout(300);
    const twin1280Png = path.join(SCREENSHOTS_DIR, "twin_top_1280.png");
    await page.screenshot({ path: twin1280Png, fullPage: false });
    console.log(`Saved Twin top viewport screenshot (1280x900): ${twin1280Png}`);
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.waitForTimeout(300);

    // 6. Capture Full Twin Page Screenshot
    const twinFullPng = path.join(SCREENSHOTS_DIR, "twin_full.png");
    await page.screenshot({ path: twinFullPng, fullPage: true });
    console.log(`Saved Twin full page screenshot: ${twinFullPng}`);

    // 7. Capture What-If Page Screenshot
    console.log("Navigating to What-If page (http://localhost:3000/what-if)...");
    await page.goto("http://localhost:3000/what-if", { waitUntil: "networkidle" });
    await page.waitForSelector("text=Physiology-Constrained What-If Simulator", { timeout: 15000 });
    await page.waitForSelector("text=Regional Dish", { timeout: 15000 });
    await page.waitForFunction(() => {
      const selects = Array.from(document.querySelectorAll("select"));
      return selects.some((s) => s.options && s.options.length >= 20);
    }, { timeout: 15000 });
    await page.waitForTimeout(800);
    const whatifPng = path.join(SCREENSHOTS_DIR, "whatif.png");
    await page.screenshot({ path: whatifPng, fullPage: false });
    console.log(`Saved What-If page screenshot: ${whatifPng}`);

    // Run real simulation in compare mode (Scenario A: 0m walk, Scenario B: 20m walk)
    console.log("Enabling Compare Mode for What-If simulation...");
    await page.evaluate(() => {
      const cbox = document.querySelector('input[type="checkbox"]');
      if (cbox && !cbox.checked) {
        cbox.click();
      }
    });
    await page.waitForTimeout(500);

    console.log("Setting Scenario B to same meal at 1x with 20m walk (and Scenario A 0m walk)...");
    await page.evaluate(() => {
      const selects = document.querySelectorAll("select");
      const foodSelects = Array.from(selects).filter((s) => s.options.length >= 20);
      if (foodSelects.length >= 2) {
        foodSelects[1].value = foodSelects[0].value;
        foodSelects[1].dispatchEvent(new Event("change", { bubbles: true }));
      }
      const sliders = document.querySelectorAll('input[type="range"]');
      if (sliders.length > 0) {
        sliders[0].value = "0";
        sliders[0].dispatchEvent(new Event("input", { bubbles: true }));
        sliders[0].dispatchEvent(new Event("change", { bubbles: true }));
      }
      if (sliders.length > 1) {
        sliders[1].value = "20";
        sliders[1].dispatchEvent(new Event("input", { bubbles: true }));
        sliders[1].dispatchEvent(new Event("change", { bubbles: true }));
      }
    });
    await page.waitForTimeout(500);

    console.log("Executing simulation via real API...");
    await page.click('button:has-text("Run What-If Simulation")');
    await page.waitForSelector("text=+30m Delta", { timeout: 20000 });
    await page.waitForTimeout(1000);

    const whatifResultPng = path.join(SCREENSHOTS_DIR, "whatif_result.png");
    await page.screenshot({ path: whatifResultPng, fullPage: false });
    console.log(`Saved What-If simulation result screenshot: ${whatifResultPng}`);

    // 8. Capture Summary Page Screenshot (viewport and full page)
    console.log("Navigating to Summary page (http://localhost:3000/summary/DEMO_PAT_001)...");
    await page.goto("http://localhost:3000/summary/DEMO_PAT_001", { waitUntil: "networkidle" });
    await page.waitForSelector("text=Standard 5-Band Time in Range Profile", { timeout: 15000 });
    await page.waitForTimeout(1000);
    const summaryPng = path.join(SCREENSHOTS_DIR, "summary.png");
    await page.screenshot({ path: summaryPng, fullPage: false });
    console.log(`Saved Summary page screenshot: ${summaryPng}`);

    const summaryFullPng = path.join(SCREENSHOTS_DIR, "summary_full.png");
    await page.screenshot({ path: summaryFullPng, fullPage: true });
    console.log(`Saved Summary full page screenshot: ${summaryFullPng}`);

    // 9. Capture Evidence Page Screenshot (viewport and full page)
    console.log("Navigating to Evidence page (http://localhost:3000/evidence)...");
    await page.goto("http://localhost:3000/evidence", { waitUntil: "networkidle" });
    await page.waitForSelector("text=Forecast Accuracy by Horizon", { timeout: 15000 });
    await page.waitForTimeout(1000);
    const evidencePng = path.join(SCREENSHOTS_DIR, "evidence.png");
    await page.screenshot({ path: evidencePng, fullPage: false });
    console.log(`Saved Evidence page screenshot: ${evidencePng}`);

    const evidenceFullPng = path.join(SCREENSHOTS_DIR, "evidence_full.png");
    await page.screenshot({ path: evidenceFullPng, fullPage: true });
    console.log(`Saved Evidence full page screenshot: ${evidenceFullPng}`);

    console.log(`Screenshot capture completed successfully! Logged ${loggedApiResponses.length} real API responses.`);
  } finally {
    if (browser) {
      console.log("Closing Playwright browser...");
      await browser.close().catch(() => {});
    }
    if (serverProcess) {
      await killServer(serverProcess);
    }
  }
}

main()
  .then(() => {
    process.exit(0);
  })
  .catch((err) => {
    console.error("Screenshot capture failed:", err);
    process.exit(1);
  });

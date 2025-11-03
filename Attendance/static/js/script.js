// ===== tiny helpers =====
const $ = (sel) => document.querySelector(sel);
const nowMs = () => Date.now();
const nowSec = () => Math.floor(Date.now() / 1000);
const randSID = () => Math.random().toString(36).slice(2, 8); // 6-char session id (kept for other uses)

// New: numeric-only 6-digit session id for Version 1 payloads
const randNumericSID = () => String(Math.floor(100000 + Math.random() * 900000)); // 6-digit numeric SID

// Persist teacher-side attendance (for manual marking / export on teacher device)
let attendance = JSON.parse(localStorage.getItem("attendance") || "{}");
function saveAttendance() { localStorage.setItem("attendance", JSON.stringify(attendance)); }
function renderAttendance() {
  const tbody = $("#attendance-table tbody");
  if (!tbody) return;
  tbody.innerHTML = "";
  Object.entries(attendance).forEach(([sid, status]) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${sid}</td><td>${status}</td>`;
    tbody.appendChild(tr);
  });
}

// Helper: pad numeric fields
const pad = (v, n) => String(v).padStart(n, "0");

// ===== TEACHER PAGE =====
if ($("#start-btn")) {
  const qrDiv = $("#qrcode");
  const meta = $("#meta");

  // Keep fallback QR generator instance (if qrcodejs exists)
  let fallbackQr = null;
  if (typeof QRCode !== "undefined") {
    // QRCode expects a container node or id; ensure qrDiv exists
    fallbackQr = new QRCode(qrDiv, { width: screen.width*0.5, height: screen.width*0.5, align: "center", colorDark: "#000000", colorLight: "#ffffff", correctLevel: QRCode.CorrectLevel.L });
  }

  let running = false;

  $("#start-btn").addEventListener("click", () => {
    if (running) return;
    running = true;

    // Use numeric SID for Version 1 packing
    const sessionId = randNumericSID();
    const total = 15;       // 30s / 2s
    const lifetimeSec = 2;  // seconds
    let seq = 0;
    let left = 30;

    meta.textContent = `Session: ${sessionId} • Time left: ${left}s`;

    const tick = () => {
      if (seq >= total) {
        running = false;
        meta.textContent = `Session: ${sessionId} • done`;
        return;
      }

      // fresh code + payload per 2 seconds
      const code = String(Math.floor(100000 + Math.random() * 900000)); // 6-digit
      const tsSec = nowSec();
      const expSec = tsSec + lifetimeSec;
       
      // Build compact numeric payload (fits QR v1 numeric capacity)
      // Format: v(1) | sid(6) | seq(2) | code(6) | ts(10) | exp(10)  => total 35 digits
      const seqStr = pad(seq, 2);
      const payloadStr = `1${sessionId}${seqStr}${code}${pad(tsSec, 10)}${pad(expSec, 10)}`;

      // If qrcode-generator is available, force TypeNumber = 1 (Version 1)
      if (typeof qrcode === "function") {
        try {
          const typeNumber = 1; // Force Version 1
          const errorLevel = "L"; // Lowest ECC -> larger modules, easier to scan far
          const qrObj = qrcode(typeNumber, errorLevel);
          qrObj.addData(payloadStr);
          qrObj.make();
          const qrSize = Math.floor(Math.min(window.innerWidth, window.innerHeight) * 0.9);
          fallbackQr.clear();
          fallbackQr._htOption.width = qrSize;
          fallbackQr._htOption.height = qrSize;
          const moduleSize = Math.floor(qrSize / 21); // 21 modules for version 1
          const imgTag = qrObj.createImgTag(moduleSize, 0);
          qrDiv.innerHTML = imgTag;

          
        } catch (e) {
          // If qrcode-generator fails for any reason, fallback to older QR library
          if (fallbackQr) fallbackQr.makeCode(payloadStr);
          else qrDiv.textContent = payloadStr; // show payload if no QR lib available
        }
      } else {
        // Fallback: existing QRCode (cannot guarantee v1)
        if (fallbackQr) fallbackQr.makeCode(payloadStr);
        else qrDiv.textContent = payloadStr;
      }

      seq++;
      left -= 2;
      meta.textContent = `Session: ${sessionId} • Time left: ${Math.max(left,0)}s`;
      setTimeout(tick, lifetimeSec * 1000);
    };
    tick();
  });

  // Manual marking (teacher-only UI, since we have no backend)
  $("#mark-present-btn").addEventListener("click", () => {
    const id = $("#manual-student-id").value.trim();
    if (!id) return;
    attendance[id] = "Present (manual)";
    saveAttendance();
    renderAttendance();
  });

  $("#export-btn").addEventListener("click", () => {
    let csv = "Student ID,Status\n";
    Object.entries(attendance).forEach(([id, st]) => { csv += `${id},${st}\n`; });
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = "attendance.csv"; a.click();
    URL.revokeObjectURL(url);
  });

  renderAttendance();
}

// ===== STUDENT PAGE =====
if ($("#qr-reader")) {
  const statusEl = $("#scan-status");
  const qrReaderDiv = $("#qr-reader");
  let scanner = null; // Declare scanner here so it's accessible globally within this block

  // Function to start the QR scanner
  function startQrScanner() {
    const myId = $("#student-id").value.trim();
    if (!myId) {
      statusEl.textContent = "Please enter and save your student id before scanning.";
      return;
    }

    if (scanner) { // If scanner already exists, clear it before re-rendering
      scanner.clear().then(() => {
        scanner = null;
        renderScanner();
      }).catch(error => {
        console.error("Failed to clear scanner: ", error);
        renderScanner(); // Try to render anyway
      });
    } else {
      renderScanner();
    }

    function renderScanner() {
      qrReaderDiv.style.display = 'block'; // Show the QR reader div
      scanner = new Html5QrcodeScanner("qr-reader", { fps: 12, qrbox: 240 });
      scanner.render(onScanSuccess, onScanError);
      statusEl.textContent = `Scanner started. Ready to scan for student ID: ${myId}`;
    }
  }

  // Remember student id locally
  $("#save-id").addEventListener("click", () => {
    const id = $("#student-id").value.trim();
    if (!id) { statusEl.textContent = "Enter a student id first."; return; }
    localStorage.setItem("student-id", id);
    statusEl.textContent = `Saved id: ${id}`;
    startQrScanner(); // Start scanner after saving ID
  });
  // restore if saved
  const sidSaved = localStorage.getItem("student-id");
  if (sidSaved) {
    $("#student-id").value = sidSaved;
    startQrScanner(); // Start scanner if ID is already saved
  } else {
    qrReaderDiv.style.display = 'none'; // Hide QR reader if no ID is saved
    statusEl.textContent = "Please enter and save your student ID to start scanning.";
  }

  // Track scans per session to ensure "two different codes"
  function getScannedSet(sessionId) {
    const raw = localStorage.getItem(`scans-${sessionId}`) || "[]";
    try { return new Set(JSON.parse(raw)); } catch { return new Set(); }
  }
  function saveScannedSet(sessionId, set) {
    localStorage.setItem(`scans-${sessionId}`, JSON.stringify([...set]));
  }

  function parseNumericPayload(s) {
    // Expected length: 35 digits
    if (!/^\d{35}$/.test(s)) return null;
    try {
      const v = Number(s.slice(0,1));
      const sid = s.slice(1,7);            // 6 digits
      const seq = Number(s.slice(7,9));    // 2 digits
      const code = s.slice(9,15);          // 6 digits
      const tsSec = Number(s.slice(15,25)); // 10 digits
      const expSec = Number(s.slice(25,35)); // 10 digits
      return { v, sid, seq, code, ts: tsSec * 1000, exp: expSec * 1000 };
    } catch (e) { return null; }
  }

  function onScanSuccess(decodedText) {
    // Try numeric v1 payload first (compact)
    let p = null;
    const numeric = parseNumericPayload(String(decodedText).trim());
    if (numeric) {
      p = numeric;
    } else {
      // Fallback: attempt JSON (older format)
      try { const parsed = JSON.parse(decodedText); if (parsed) p = parsed; } catch { /* ignore */ }
    }

    if (!p) {
      statusEl.textContent = "Invalid QR payload.";
      return;
    }

    // Validate structure (support both formats)
    // JSON format used ms (as original); numeric format we converted ts/exp to ms above.
    if (!(p && (p.v === 1) && typeof p.sid === "string" && typeof p.seq === "number" &&
          (/^\d{6}$/.test(String(p.code)) || typeof p.code === "number") &&
          typeof p.ts === "number" && typeof p.exp === "number")) {
      statusEl.textContent = "Malformed QR.";
      return;
    }

    // Validate expiry (2s window, allow small skew)
    const now = nowMs();
    const skewMs = 3000; // ms tolerance
    if (now > (p.exp + skewMs)) {
      statusEl.textContent = "Expired QR – try the next one.";
      return;
    }

    // Ensure two different codes (distinct seq) within the same session
    const myId = $("#student-id").value.trim() || localStorage.getItem("student-id") || "";
    if (!myId) { statusEl.textContent = "Please enter and save your student id."; return; }

    const scanned = getScannedSet(p.sid);
    const before = scanned.size;
    scanned.add(p.seq);     // uniqueness based on sequence number
    saveScannedSet(p.sid, scanned);

    const scans = scanned.size;
    if (scans >= 2) {
      statusEl.textContent = `✅ Marked PRESENT (id=${myId}). You scanned ${scans} codes in session ${p.sid}.`;
      // Send attendance to backend
      fetch("/submit_student_qr_attendance", {
        method: "POST",
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ student_id: myId, session_id: p.sid })
      }).then(response => response.json())
        .then(data => {
          console.log("Backend response for attendance submission:", data);
          if (data.success) {
            console.log("Attendance submitted successfully:", data.message);
            statusEl.textContent = `✅ ${data.message}`;
            if (scanner) {
              console.log("Attempting to clear scanner...");
              scanner.clear().then(() => {
                console.log("Scanner cleared successfully. Hiding QR reader div.");
                qrReaderDiv.style.display = 'none'; // Hide the QR reader div after scanner clears
              }).catch(error => {
                console.error("Failed to clear scanner after successful attendance: ", error);
                qrReaderDiv.style.display = 'none'; // Hide anyway if clear fails
              });
            } else {
              console.warn("Scanner object is null or undefined when trying to clear.");
              qrReaderDiv.style.display = 'none'; // Hide anyway
            }
          } else {
            console.error("Error submitting attendance:", data.message);
            statusEl.textContent = `Error: ${data.message}`;
          }
        })
        .catch(error => {
          console.error("Network error submitting attendance:", error);
          statusEl.textContent = "Network error. Please try again.";
        });
      // (Optional) keep a local note for the student
      localStorage.setItem(`present-${p.sid}-${myId}`, "true");
    } else if (scanned.size > before) {
      statusEl.textContent = `Good! 1/2 scans recorded. Scan the next code…`;
    } else {
      statusEl.textContent = `Already counted this code. Scan the next one.`;
    }
  }

  function onScanError() { /* ignore noisy frames */ }

  // Initial state: hide QR reader if no ID is saved
  if (!sidSaved) {
    qrReaderDiv.style.display = 'none';
  }
}

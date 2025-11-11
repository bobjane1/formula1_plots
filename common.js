// Common utilities for F1 data visualization pages

// Cache management
const DataCache = {
  // Storage keys
  CACHE_BUST_KEY: 'f1_data_cache_bust',
  SESSION_VERSION_KEY: 'f1_session_version',

  // Cache of bust parameter for this page load
  _sessionVersion: null,
  _sessionVersionInitialized: false,

  // Get or create session version (persists for entire browser session)
  getSessionVersion() {
    if (!this._sessionVersionInitialized) {
      // Check if we have a reload flag set
      const reloadFlag = sessionStorage.getItem(this.CACHE_BUST_KEY);
      if (reloadFlag) {
        // Reload was requested - create new session version
        this._sessionVersion = reloadFlag;
        sessionStorage.setItem(this.SESSION_VERSION_KEY, reloadFlag);
        sessionStorage.removeItem(this.CACHE_BUST_KEY);
      } else {
        // Check if we already have a session version
        this._sessionVersion = sessionStorage.getItem(this.SESSION_VERSION_KEY);
      }
      this._sessionVersionInitialized = true;
    }
    return this._sessionVersion;
  },

  // Check if we should bust cache
  shouldBustCache() {
    return this.getSessionVersion();
  },

  // Get CSV URL - add cache buster if session version exists
  getCSVUrl(filename) {
    const sessionVersion = this.getSessionVersion();
    if (sessionVersion) {
      const url = `${filename}?v=${sessionVersion}`;
      return url;
    }
    // No query parameter - let browser use normal HTTP caching
    return filename;
  },

  // Reload data (invalidate cache)
  reloadData() {
    // Set a flag with timestamp that will be used on next page load
    const timestamp = Date.now().toString();
    sessionStorage.setItem(this.CACHE_BUST_KEY, timestamp);
    window.location.reload();
  },

  // Wrap PapaParse to add timing
  loadCSV(url, config) {
    const startTime = performance.now();

    const originalComplete = config.complete;
    config.complete = function(results) {
      const loadTime = (performance.now() - startTime).toFixed(0);
      if (originalComplete) {
        originalComplete(results);
      }
    };

    Papa.parse(url, config);
  }
};

// Parsed CSV data cache - stores parsed data in IndexedDB to avoid re-parsing
const ParsedDataCache = {
  dbName: 'F1ParsedDataCache',
  storeName: 'parsedCSV',
  version: 1,
  db: null,

  // Initialize IndexedDB
  async init() {
    if (this.db) return this.db;

    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.dbName, this.version);

      request.onerror = () => {
        console.error('[ParsedDataCache] IndexedDB error:', request.error);
        reject(request.error);
      };

      request.onsuccess = () => {
        this.db = request.result;
        resolve(this.db);
      };

      request.onupgradeneeded = (event) => {
        const db = event.target.result;
        if (!db.objectStoreNames.contains(this.storeName)) {
          db.createObjectStore(this.storeName, { keyPath: 'filename' });
        }
      };
    });
  },

  // Get parsed data from cache
  async get(filename) {
    try {
      await this.init();
      return new Promise((resolve, reject) => {
        const transaction = this.db.transaction([this.storeName], 'readonly');
        const store = transaction.objectStore(this.storeName);
        const request = store.get(filename);

        request.onsuccess = () => {
          if (request.result) {
            resolve(request.result.results);
          } else {
            resolve(null);
          }
        };

        request.onerror = () => {
          console.error('[ParsedDataCache] Error reading from IndexedDB:', request.error);
          resolve(null);
        };
      });
    } catch (e) {
      console.error('[ParsedDataCache] Error accessing IndexedDB:', e);
      return null;
    }
  },

  // Store parsed data in cache
  async set(filename, results) {
    try {
      await this.init();
      return new Promise((resolve, reject) => {
        const transaction = this.db.transaction([this.storeName], 'readwrite');
        const store = transaction.objectStore(this.storeName);
        const data = {
          filename: filename,
          results: results,
          rows: results.data ? results.data.length : 0,
          timestamp: Date.now()
        };
        const request = store.put(data);

        request.onsuccess = () => {
          resolve();
        };

        request.onerror = () => {
          console.error('[ParsedDataCache] Error writing to IndexedDB:', request.error);
          resolve();
        };
      });
    } catch (e) {
      console.error('[ParsedDataCache] Error accessing IndexedDB:', e);
    }
  },

  // Clear all parsed data caches
  async clear() {
    try {
      await this.init();
      return new Promise((resolve, reject) => {
        const transaction = this.db.transaction([this.storeName], 'readwrite');
        const store = transaction.objectStore(this.storeName);
        const request = store.clear();

        request.onsuccess = () => {
          resolve();
        };

        request.onerror = () => {
          console.error('[ParsedDataCache] Error clearing IndexedDB:', request.error);
          resolve();
        };
      });
    } catch (e) {
      console.error('[ParsedDataCache] Error accessing IndexedDB:', e);
    }
  },

  // Load CSV with caching - wrapper around Papa.parse
  loadCSV(filename, config) {
    // Check if we should bust cache
    const shouldBust = DataCache.shouldBustCache();

    // Try to get from cache if not busting
    if (!shouldBust) {
      this.get(filename).then(cached => {
        if (cached) {
          // Use cached data
          if (config.complete) {
            config.complete(cached);
          }
        } else {
          // Not in cache - load and parse
          this._loadAndParse(filename, config);
        }
      }).catch(() => {
        // Error accessing cache - just load and parse
        this._loadAndParse(filename, config);
      });
    } else {
      // Cache bust requested - load and parse
      this._loadAndParse(filename, config);
    }
  },

  // Internal method to load and parse CSV
  _loadAndParse(filename, config) {
    const url = DataCache.getCSVUrl(filename);

    const originalComplete = config.complete;
    config.complete = (results) => {
      // Store in cache (async, don't wait)
      this.set(filename, results);

      // Call original callback
      if (originalComplete) {
        originalComplete(results);
      }
    };

    Papa.parse(url, config);
  }
};

// Clear parsed data cache when reload is requested
const originalReloadData = DataCache.reloadData;
DataCache.reloadData = function() {
  ParsedDataCache.clear();
  originalReloadData.call(this);
};

// No initialization needed - we use sessionStorage which is cleared between browser sessions

// Create top navigation dynamically
function createTopNav() {
  // Check if nav already exists (manually created)
  if (document.querySelector('nav.top-nav')) return;

  // Get current page from data attribute or infer from URL
  const currentPage = document.body.dataset.page || inferCurrentPage();

  // Create nav element
  const nav = document.createElement('nav');
  nav.className = 'top-nav';

  // Define navigation links
  const links = [
    { href: 'strategy.html', text: 'Strategy', page: 'strategy' },
    { href: 'race_trace.html', text: 'Race Trace', page: 'race_trace' },
    { href: 'tyre_deg.html', text: 'Tyre Deg', page: 'tyre_deg' },
    { href: 'calculator.html', text: 'Calculator', page: 'calculator' }
  ];

  // Create links
  links.forEach(link => {
    const a = document.createElement('a');
    a.href = link.href;
    a.textContent = link.text;
    if (link.page === currentPage) {
      a.className = 'active';
    }
    nav.appendChild(a);
  });

  // Insert at beginning of body
  document.body.insertBefore(nav, document.body.firstChild);
}

// Infer current page from URL
function inferCurrentPage() {
  const path = window.location.pathname;
  const filename = path.split('/').pop().replace('.html', '');
  return filename || 'strategy';
}

// Add reload data button to navigation
function addReloadDataButton() {
  const nav = document.querySelector('nav');
  if (!nav) return;

  // Check if button already exists
  if (document.getElementById('reload-data-btn')) return;

  // Create button with dark styling
  const button = document.createElement('button');
  button.id = 'reload-data-btn';
  button.textContent = '↻ Reload Data';
  button.style.cssText = `
    background: #11141a;
    color: #8b8b8b;
    text-decoration: none;
    padding: 8px 16px;
    font-size: 13px;
    font-weight: 500;
    border: 1px solid #2a2f3a;
    border-radius: 8px;
    cursor: pointer;
    margin-left: auto;
    margin-top: 8px;
    margin-bottom: 8px;
    margin-right: 12px;
    transition: all 0.15s ease;
  `;

  button.onmouseover = () => {
    button.style.background = '#1a1d24';
    button.style.color = '#e8e8e8';
    button.style.borderColor = '#00e0ff';
  };
  button.onmouseout = () => {
    button.style.background = '#11141a';
    button.style.color = '#8b8b8b';
    button.style.borderColor = '#2a2f3a';
  };
  button.onclick = () => {
    DataCache.reloadData();
  };

  // Add to nav (append at end)
  nav.appendChild(button);
}

// Initialize navigation on DOM load
function initNavigation() {
  createTopNav();
  addReloadDataButton();
}

// Initialize on DOM load
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initNavigation);
} else {
  initNavigation();
}

// Common UI utilities
const UIHelpers = {
  // Populate a select element with unique, non-empty values
  // Clears existing options and adds new ones
  populateSelect(selectElement, values, options = {}) {
    const {
      filterEmpty = true,
      defaultValue = null,
      autoSelectFirst = true,
      autoSelectValue = null,
      onChange = null
    } = options;

    // Clear existing options
    selectElement.innerHTML = '';

    // Filter and get unique values
    let filteredValues = values;
    if (filterEmpty) {
      filteredValues = values.filter(v => v && String(v).trim() !== '');
    }
    const uniqueValues = [...new Set(filteredValues)];

    // Add options
    uniqueValues.forEach(value => {
      const opt = document.createElement('option');
      opt.value = value;
      opt.textContent = value;
      selectElement.appendChild(opt);
    });

    // Enable the select if there are options
    selectElement.disabled = uniqueValues.length === 0;

    // Auto-select logic
    if (uniqueValues.length > 0) {
      if (autoSelectValue && uniqueValues.includes(autoSelectValue)) {
        selectElement.value = autoSelectValue;
        if (onChange) onChange();
      } else if (autoSelectFirst) {
        selectElement.value = uniqueValues[0];
        if (onChange) onChange();
      }
    }

    return uniqueValues;
  },

  // Populate seasons from sessionsByKey Map
  // Selects most recent season by default
  populateSeasons(selectElement, sessionsByKey, options = {}) {
    const { onChange = null } = options;

    const seasons = new Set();
    sessionsByKey.forEach((val) => {
      if (val.year) {
        seasons.add(val.year);
      }
    });

    const sortedSeasons = Array.from(seasons).sort((a, b) => {
      return parseInt(b, 10) - parseInt(a, 10); // Descending (most recent first)
    });

    selectElement.innerHTML = '';
    sortedSeasons.forEach(year => {
      const opt = document.createElement('option');
      opt.value = year;
      opt.textContent = year;
      selectElement.appendChild(opt);
    });

    selectElement.disabled = sortedSeasons.length === 0;

    // Select most recent season by default
    if (sortedSeasons.length > 0) {
      selectElement.value = sortedSeasons[0];
      if (onChange) onChange();
    }

    return sortedSeasons;
  },

  // Populate GPs for a given season from sessionsByKey Map
  // Selects last GP (highest round number) by default
  populateGPs(selectElement, sessionsByKey, selectedSeason, options = {}) {
    const { onChange = null } = options;

    selectElement.innerHTML = '';
    selectElement.disabled = true;

    if (!selectedSeason) return [];

    const gps = new Map();
    sessionsByKey.forEach((val) => {
      if (val.year === selectedSeason) {
        const gpKey = `${val.round_no}||${val.meeting_name}`;
        if (!gps.has(gpKey)) {
          gps.set(gpKey, {
            round_no: val.round_no,
            meeting_name: val.meeting_name
          });
        }
      }
    });

    const sortedGPs = Array.from(gps.values()).sort((a, b) => {
      const ra = parseInt(a.round_no, 10) || 0;
      const rb = parseInt(b.round_no, 10) || 0;
      return ra - rb; // Ascending order
    });

    sortedGPs.forEach(gp => {
      const opt = document.createElement('option');
      opt.value = `${gp.round_no}||${gp.meeting_name}`;
      opt.textContent = `${gp.round_no}. ${gp.meeting_name}`;
      selectElement.appendChild(opt);
    });

    selectElement.disabled = sortedGPs.length === 0;

    // Select last GP by default
    if (sortedGPs.length > 0) {
      const lastGP = sortedGPs[sortedGPs.length - 1];
      selectElement.value = `${lastGP.round_no}||${lastGP.meeting_name}`;
      if (onChange) onChange();
    }

    return sortedGPs;
  },

  // Populate sessions for a given season and GP from sessionsByKey Map
  // Prefers 'R' (Race) by default, otherwise selects first
  populateSessions(selectElement, sessionsByKey, selectedSeason, selectedGP, options = {}) {
    const { onChange = null, preferredSession = 'R' } = options;

    selectElement.innerHTML = '';
    selectElement.disabled = true;

    if (!selectedGP) return [];

    const [round, meeting] = selectedGP.split('||');

    const sessions = [];
    sessionsByKey.forEach((val) => {
      if (val.year === selectedSeason && val.round_no === round && val.meeting_name === meeting) {
        sessions.push(val.session_type);
      }
    });

    return this.populateSelect(selectElement, sessions, {
      filterEmpty: true,
      autoSelectValue: preferredSession,
      autoSelectFirst: true,
      onChange: onChange
    });
  }
};

// Driver info management
const DriverInfo = {
  // Storage for driver info: key: "year|round|session" -> Map(driver -> {fullName, color, teamName, headshotUrl})
  _driverInfoBySession: new Map(),
  _loaded: false,
  _loading: false,
  _callbacks: [],

  // Load driver info CSV
  load(callback) {
    // If already loaded, call callback immediately
    if (this._loaded) {
      if (callback) callback();
      return;
    }

    // If currently loading, queue callback
    if (this._loading) {
      if (callback) this._callbacks.push(callback);
      return;
    }

    // Start loading
    this._loading = true;
    if (callback) this._callbacks.push(callback);

    Papa.parse(DataCache.getCSVUrl("driver_info.csv"), {
      download: true,
      header: true,
      skipEmptyLines: true,
      dynamicTyping: false,
      complete: (results) => {
        if (results.data) {
          this._driverInfoBySession.clear();

          results.data.forEach(row => {
            const year = String(row.year || "").trim();
            const round = parseInt(row.round_no, 10);
            const driver = String(row.driver || "").trim();
            const session = String(row.session_type || "").trim().toUpperCase();
            const colorRaw = String(row.color || "").trim();
            const colorOk = /^#?[0-9a-f]{6}$/i.test(colorRaw.replace('#',''));
            const color = colorOk ? (colorRaw.startsWith('#') ? colorRaw : ('#' + colorRaw)) : null;
            const teamName = String(row.TeamName || "").trim();
            const fullName = String(row.FullName || "").trim();
            const headshotUrl = String(row.HeadshotUrl || "").trim();

            if (!year || !Number.isFinite(round) || !driver) return;

            // Store by base key (year|round)
            const kBase = `${year}|${round}`;
            if (!this._driverInfoBySession.has(kBase)) {
              this._driverInfoBySession.set(kBase, new Map());
            }
            const baseMap = this._driverInfoBySession.get(kBase);
            const existing = baseMap.get(driver) || {};
            baseMap.set(driver, {
              color: color ?? existing.color ?? null,
              fullName: fullName || existing.fullName || '',
              teamName: teamName || existing.teamName || '',
              headshotUrl: headshotUrl || existing.headshotUrl || ''
            });

            // Also store by session-specific key (year|round|session)
            if (session) {
              const kSess = `${kBase}|${session}`;
              if (!this._driverInfoBySession.has(kSess)) {
                this._driverInfoBySession.set(kSess, new Map());
              }
              const sessMap = this._driverInfoBySession.get(kSess);
              const ex2 = sessMap.get(driver) || {};
              sessMap.set(driver, {
                color: color ?? ex2.color ?? null,
                fullName: fullName || ex2.fullName || '',
                teamName: teamName || ex2.teamName || '',
                headshotUrl: headshotUrl || ex2.headshotUrl || ''
              });
            }
          });
        }

        this._loaded = true;
        this._loading = false;

        // Call all queued callbacks
        this._callbacks.forEach(cb => cb());
        this._callbacks = [];
      },
      error: (err) => {
        console.error("Error loading driver_info.csv:", err);
        this._loaded = true;
        this._loading = false;

        // Call all queued callbacks even on error
        this._callbacks.forEach(cb => cb());
        this._callbacks = [];
      }
    });
  },

  // Get driver info for a specific session
  // Returns { fullName, color, teamName, headshotUrl }
  getDriverInfo(driver, year, round, sessionType) {
    if (!this._loaded) {
      console.warn("DriverInfo.getDriverInfo called before data was loaded");
      return { fullName: driver, color: '#888888', teamName: '', headshotUrl: '' };
    }

    const sessionUpper = String(sessionType || "").trim().toUpperCase();

    // Try session-specific key first
    const kSess = `${year}|${round}|${sessionUpper}`;
    if (this._driverInfoBySession.has(kSess)) {
      const sessMap = this._driverInfoBySession.get(kSess);
      if (sessMap.has(driver)) {
        const info = sessMap.get(driver);
        return {
          fullName: info.fullName || driver,
          color: info.color || '#888888',
          teamName: info.teamName || '',
          headshotUrl: info.headshotUrl || ''
        };
      }
    }

    // Fall back to base key (year|round)
    const kBase = `${year}|${round}`;
    if (this._driverInfoBySession.has(kBase)) {
      const baseMap = this._driverInfoBySession.get(kBase);
      if (baseMap.has(driver)) {
        const info = baseMap.get(driver);
        return {
          fullName: info.fullName || driver,
          color: info.color || '#888888',
          teamName: info.teamName || '',
          headshotUrl: info.headshotUrl || ''
        };
      }
    }

    // Final fallback
    return { fullName: driver, color: '#888888', teamName: '', headshotUrl: '' };
  },

  // Get driver colors for a specific session as a Map(driver -> color)
  getDriverColors(year, round, sessionType) {
    const colorMap = new Map();

    if (!this._loaded) {
      console.warn("DriverInfo.getDriverColors called before data was loaded");
      return colorMap;
    }

    const sessionUpper = String(sessionType || "").trim().toUpperCase();

    // Try session-specific key first
    const kSess = `${year}|${round}|${sessionUpper}`;
    if (this._driverInfoBySession.has(kSess)) {
      const sessMap = this._driverInfoBySession.get(kSess);
      sessMap.forEach((info, driver) => {
        if (info.color) {
          colorMap.set(driver, info.color);
        }
      });
      return colorMap;
    }

    // Fall back to base key (year|round)
    const kBase = `${year}|${round}`;
    if (this._driverInfoBySession.has(kBase)) {
      const baseMap = this._driverInfoBySession.get(kBase);
      baseMap.forEach((info, driver) => {
        if (info.color) {
          colorMap.set(driver, info.color);
        }
      });
    }

    return colorMap;
  }
};

// Export for use in other scripts
window.DataCache = DataCache;
window.ParsedDataCache = ParsedDataCache;
window.UIHelpers = UIHelpers;
window.DriverInfo = DriverInfo;

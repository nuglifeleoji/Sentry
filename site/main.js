function initializeTheme() {
  const controls = document.querySelector("[data-theme-controls]");
  if (!controls) return;

  const buttons = Array.from(controls.querySelectorAll("[data-theme-choice]"));
  const root = document.documentElement;
  const storageKey = "sentry-site-theme";
  let theme = root.dataset.theme;

  try {
    const saved = localStorage.getItem(storageKey);
    if (buttons.some((button) => button.dataset.themeChoice === saved)) {
      theme = saved;
    }
  } catch {
    // A blocked storage API only disables persistence.
  }

  function applyTheme(choice) {
    root.dataset.theme = choice;
    for (const button of buttons) {
      button.setAttribute("aria-pressed", String(button.dataset.themeChoice === choice));
    }
  }

  applyTheme(theme);
  for (const button of buttons) {
    button.addEventListener("click", () => {
      const choice = button.dataset.themeChoice;
      applyTheme(choice);
      try {
        localStorage.setItem(storageKey, choice);
      } catch {
        // Keep the selected theme for this page even when it cannot be saved.
      }
    });
  }
  controls.hidden = false;
}

function initializeBenchmarkTabs() {
  const tabList = document.querySelector("[data-benchmark-tabs]");
  if (!tabList) return;

  const tabs = Array.from(tabList.querySelectorAll("[data-result-tab]"));
  const panels = tabs.map((tab) => document.getElementById(tab.getAttribute("aria-controls")));
  if (!tabs.length || panels.some((panel) => !panel?.matches("[data-result-panel]"))) return;

  function selectTab(activeIndex) {
    tabs.forEach((tab, index) => {
      const selected = index === activeIndex;
      tab.setAttribute("aria-selected", String(selected));
      tab.tabIndex = selected ? 0 : -1;
      panels[index].hidden = !selected;
    });
  }

  tabList.setAttribute("role", "tablist");
  tabs.forEach((tab, index) => {
    tab.setAttribute("role", "tab");
    panels[index].setAttribute("role", "tabpanel");
    panels[index].setAttribute("aria-labelledby", tab.id);
    panels[index].tabIndex = 0;
    tab.addEventListener("click", () => selectTab(index));
    tab.addEventListener("keydown", (event) => {
      let nextIndex;
      switch (event.key) {
        case "ArrowLeft":
          nextIndex = (index - 1 + tabs.length) % tabs.length;
          break;
        case "ArrowRight":
          nextIndex = (index + 1) % tabs.length;
          break;
        case "Home":
          nextIndex = 0;
          break;
        case "End":
          nextIndex = tabs.length - 1;
          break;
        default:
          return;
      }
      event.preventDefault();
      selectTab(nextIndex);
      tabs[nextIndex].focus();
    });
  });

  selectTab(0);
  tabList.hidden = false;
}

function initializeCitationCopy() {
  const button = document.querySelector("[data-copy-citation]");
  const citation = document.querySelector("[data-citation]");
  const status = document.querySelector("[data-copy-status]");
  if (!button || !citation || !status) return;

  button.addEventListener("click", async () => {
    status.hidden = false;
    status.textContent = "";
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(citation.textContent.trim());
        status.textContent = "Citation copied.";
        return;
      }
    } catch {
      // The API can be available even when permission to write is denied.
    }
    status.textContent = "Clipboard unavailable. Select the BibTeX text and copy it manually.";
  });
  button.hidden = false;
}

initializeTheme();
initializeBenchmarkTabs();
initializeCitationCopy();

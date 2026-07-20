/**
 * MedPredict Clinic Assessment Portal
 * Custom interactivity for disease-tab selection, loading animation,
 * mobile navigation drawer toggles, and asynchronous health insights.
 */

document.addEventListener("DOMContentLoaded", function () {
    const tabs = document.querySelectorAll(".disease-tab");
    const diseaseInput = document.getElementById("disease-input");
    const predictionForm = document.getElementById("prediction-form");

    // Mobile Navigation Menu Elements
    const mobileMenuToggle = document.getElementById("mobile-menu-toggle");
    const sidebar = document.getElementById("sidebar");
    const sidebarOverlay = document.getElementById("sidebar-overlay");

    function openMobileMenu() {
        if (sidebar) sidebar.classList.add("open");
        if (sidebarOverlay) sidebarOverlay.classList.add("active");
        if (mobileMenuToggle) mobileMenuToggle.setAttribute("aria-expanded", "true");
    }

    function closeMobileMenu() {
        if (sidebar) sidebar.classList.remove("open");
        if (sidebarOverlay) sidebarOverlay.classList.remove("active");
        if (mobileMenuToggle) mobileMenuToggle.setAttribute("aria-expanded", "false");
    }

    if (mobileMenuToggle) {
        mobileMenuToggle.addEventListener("click", function (e) {
            e.stopPropagation();
            if (sidebar && sidebar.classList.contains("open")) {
                closeMobileMenu();
            } else {
                openMobileMenu();
            }
        });
    }

    if (sidebarOverlay) {
        sidebarOverlay.addEventListener("click", closeMobileMenu);
    }

    // Toggles active metrics entry fields in the form
    function toggleSections(selectedDisease) {
        let sections = ["diabetes", "heart", "parkinsons", "breast"];

        sections.forEach(id => {
            let div = document.getElementById(id + "-inputs");
            if (div) {
                div.style.display = "none";
                div.querySelectorAll("input").forEach(inp => {
                    inp.disabled = true;
                });
            }
        });

        let activeDiv = document.getElementById(selectedDisease + "-inputs");
        if (activeDiv) {
            activeDiv.style.display = "block";
            activeDiv.querySelectorAll("input").forEach(inp => {
                inp.disabled = false;
            });
        }
    }

    // Bind click handler for condition tabs
    tabs.forEach(tab => {
        tab.addEventListener("click", function () {
            const target = this.getAttribute("data-target");
            if (diseaseInput) {
                diseaseInput.value = target;
            }

            tabs.forEach(t => t.classList.remove("active"));
            this.classList.add("active");

            toggleSections(target);

            // Automatically close mobile navigation drawer on selection
            closeMobileMenu();
        });
    });

    if (diseaseInput) {
        toggleSections(diseaseInput.value);
    }

    if (predictionForm) {
        predictionForm.addEventListener("submit", function () {
            const submitBtn = predictionForm.querySelector(".predict-submit-btn");
            if (submitBtn) {
                submitBtn.innerHTML = `
                    <span>Analyzing...</span> <i class="fa-solid fa-circle-notch fa-spin"></i>
                `;
                setTimeout(() => {
                    submitBtn.disabled = true;
                    submitBtn.style.opacity = "0.7";
                    submitBtn.style.cursor = "not-allowed";
                }, 5);
            }
        });
    }
});
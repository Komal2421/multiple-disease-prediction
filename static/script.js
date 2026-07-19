document.addEventListener("DOMContentLoaded", function () {
    const tabs = document.querySelectorAll(".disease-tab");
    const diseaseInput = document.getElementById("disease-input");
    const predictionForm = document.getElementById("prediction-form");

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

    tabs.forEach(tab => {
        tab.addEventListener("click", function() {
            const target = this.getAttribute("data-target");
            if (diseaseInput) {
                diseaseInput.value = target;
            }
            
            tabs.forEach(t => t.classList.remove("active"));
            this.classList.add("active");
            
            toggleSections(target);
        });
    });

    if (diseaseInput) {
        toggleSections(diseaseInput.value);
    }

    if (predictionForm) {
        predictionForm.addEventListener("submit", function() {
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
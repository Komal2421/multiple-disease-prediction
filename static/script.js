 document.addEventListener("DOMContentLoaded", function () {

    const diseaseSelect = document.getElementById("disease");

    function toggleSections(selected) {

        let sections = ["diabetes", "heart", "parkinsons", "breast"];

        sections.forEach(id => {
            let div = document.getElementById(id);

            if (div) {
                div.style.display = "none";

                div.querySelectorAll("input").forEach(inp => {
                    inp.disabled = true;
                });
            }
        });

        let active = document.getElementById(selected);

        if (active) {
            active.style.display = "block";

            active.querySelectorAll("input").forEach(inp => {
                inp.disabled = false;
            });
        }
    }

    diseaseSelect.addEventListener("change", function () {
        toggleSections(this.value);
    });

    // default load
    toggleSections(diseaseSelect.value);

});
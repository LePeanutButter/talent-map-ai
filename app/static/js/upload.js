function escapeHtml(str) {
    if (!str && str !== "") return "";
    return String(str)
        .replaceAll('&', "&amp;")
        .replaceAll('<', "&lt;")
        .replaceAll('>', "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll('\'', "&#039;");
}

$(function() {
    $("#upload-form").on("submit", function(e) {
        e.preventDefault();

        let formData = new FormData();

        const selectedJobs = $("#job-select").val();
        if (!selectedJobs || selectedJobs.length === 0) {
            alert("Please select at least one job offer first.");
            return;
        }

        const fullOffers = selectedJobs.map(id => JOB_OFFERS[id]);
        formData.append("job_text", fullOffers.join("\n\n---\n\n"));

        const files = $("input[name='file']")[0].files;
        if (files.length === 0) {
            alert("Please select at least one file.");
            return;
        }

        for (const element of files) {
            formData.append("file", element);
        }

        $("#result").html("<p><em>Uploading and processing... Please wait.</em></p>");

        $.ajax({
            url: "/api/resume/",
            method: "POST",
            data: formData,
            contentType: false,
            processData: false,
            success: function(res) {
                let html = "<h4>Results:</h4>";

                for (const [i, txt] of res.extracted_texts.entries()) {
                    let score = Number.parseFloat(res.prediction_scores[i]);
                    let percent = score * 100;
                    if (percent < 0) percent = 0;
                    percent = percent.toFixed(2);

                    const fileName = files[i] ? files[i].name : "Unknown file";
                    const arrowSvg = `<img src="/static/svg/dropdown-arrow-svgrepo-com.svg" alt="Toggle" class="arrow-icon" />`;

                    html += `
                        <hr>
                        <div class="file-block">
                            <div class="match-row" data-index="${i}" role="button" aria-expanded="false">
                                <div class="file-info">
                                    <strong class="file-label">File:</strong> <span class="file-name">${fileName}</span>
                                </div>
                                <div class="match-info">
                                    <strong>Match percentage:</strong> <span class="match-percent">${percent}%</span>
                                </div>
                                <div class="match-arrow">${arrowSvg}</div>
                            </div>
                            <div class="match-panel" id="panel-${i}">
                                <div class="match-score">Prediction Score: ${score}</div>
                                <div class="match-text">${escapeHtml(txt)}</div>
                            </div>
                        </div>
                        `;
                    }

                    $("#result").html(html);

                    $("#result").off("click", ".match-row").on("click", ".match-row", function() {
                    const $row = $(this);
                    const idx = $row.data("index");
                    const $panel = $("#panel-" + idx);

                    const isOpen = $row.hasClass("open");
                    if (isOpen) {
                        $row.removeClass("open").attr("aria-expanded", "false");
                        $panel.removeClass("open");
                    } else {
                        $row.addClass("open").attr("aria-expanded", "true");
                        $panel.addClass("open");
                    }
                });
            },
            error: function(xhr) {
                $("#result").html(
                    "<p class='text-danger'><strong>Error:</strong> " + xhr.responseText + "</p>"
                );
            }
        });
    });
});

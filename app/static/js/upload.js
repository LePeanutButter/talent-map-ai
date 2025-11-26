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
                res.extracted_texts.forEach((txt, i) => {
                    html += `<hr><h5>File ${i + 1}</h5>
                             <p><strong>Prediction Score:</strong> ${res.prediction_scores[i]}</p>
                             <pre>${txt}</pre>`;
                });
                $("#result").html(html);
            },
            error: function(xhr) {
                $("#result").html(
                    "<p class='text-danger'><strong>Error:</strong> " + xhr.responseText + "</p>"
                );
            }
        });
    });
});
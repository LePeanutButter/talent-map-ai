$(document).ready(function() {
    const select = $("#job-select");

    for (const [id, fullText] of Object.entries(JOB_OFFERS)) {
        const label = fullText.trim().split("\n")[0];
        select.append(new Option(label, id));
    }
});

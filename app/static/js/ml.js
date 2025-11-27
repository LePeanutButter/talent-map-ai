let modelStatusInterval = setInterval(checkModelStatus, 2000);

function checkModelStatus() {
    $.get("/ml/status/", function(data) {
        if (data.status === "training") {
            $("#ml-status-box")
                .removeClass()
                .addClass("alert alert-warning")
                .html("<strong>Model is training...</strong> Please wait.");
            $("#submit-btn").prop("disabled", true);
        }
        else if (data.status === "ready") {
            $("#ml-status-box")
                .removeClass()
                .addClass("alert alert-success")
                .html("<strong>ML Model Ready!</strong> You can upload CVs now.");
            $("#submit-btn").prop("disabled", false);

            clearInterval(modelStatusInterval);
        }
    }).fail(function() {
        $("#ml-status-box")
            .removeClass()
            .addClass("alert alert-danger")
            .html("<strong>Error:</strong> Could not fetch model status.");
    });
}

checkModelStatus();
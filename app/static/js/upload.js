$(function() {
    $('#upload-form').on('submit', function(e) {
        e.preventDefault();

        const fileInput = $(this).find('input[name="file"]')[0];
        const file = fileInput.files[0];
        const $result = $('#result');
        $result.empty();

        if (!file) {
            $result.html('<div class="alert alert-warning">Por favor, selecciona un archivo.</div>');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        $result.html('<div class="spinner-border text-primary" role="status"><span class="sr-only">Cargando...</span></div>');

        $.ajax({
            url: '/api/resume/',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(data) {
                $result.html(`<pre>${data.text || JSON.stringify(data, null, 2)}</pre>`);
            },
            error: function(xhr) {
                const errorMsg = xhr.responseJSON?.error || 'Ocurrió un error';
                $result.html(`<div class="alert alert-danger">${errorMsg}</div>`);
            }
        });
    });
});
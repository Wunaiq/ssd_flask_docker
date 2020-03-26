$(document).ready(function () {
    // Init
    // $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();
    $('.detection-result').hide();

    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650); 
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').text('');
        $('#result').hide();
        readURL(this);
    });

    // Predict
    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);

        // Show loading animation
        $(this).hide();
        $('.loader').show();

        
        // Make prediction by calling api /predictResNet50
        $.ajax(
            {
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            datatype: "json",
            success: function (response) {
                // Get and display the result
                $('.loader').hide();
                $('.detection-result').show();
                $('#result-img').attr('src', response);
                console.log(response);
            },
            error:function(data){
                alert("响应失败！");
            },
        });
        



    }
    
    
    );

});
document.addEventListener("DOMContentLoaded", function() {
    const fileInput = document.getElementById("fileInput");
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");

    fileInput.addEventListener("change", function(e) {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                const img = new Image();
                img.onload = function() {
                    // Draw the image on canvas
                    canvas.width = img.width;
                    canvas.height = img.height;
                    ctx.drawImage(img, 0, 0);

                    // Convert to black and white
                    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                    for (let i = 0; i < imageData.data.length; i += 4) {
                        const avg = (imageData.data[i] + imageData.data[i + 1] + imageData.data[i + 2]) / 3;
                        imageData.data[i] = 255 - avg; // Invert pixel values
                        imageData.data[i + 1] = 255 - avg;
                        imageData.data[i + 2] = 255 - avg;
                    }
                    ctx.putImageData(imageData, 0, 0);
                };
                img.src = e.target.result;
            };
            reader.readAsDataURL(file);
        }
    });
});

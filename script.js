// Wait for the DOM to fully load before running the script
document.addEventListener('DOMContentLoaded', async () => {
  const imageUpload = document.getElementById('imageUpload');
  let image, canvas, faceMatcher;

  // Load the models
  await Promise.all([
    faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
    faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
    faceapi.nets.ssdMobilenetv1.loadFromUri('/models')
  ]);

  console.log("Models loaded");

  // Load labeled face descriptors and initialize faceMatcher
  const labeledDescriptors = await loadLabeledImages();
  faceMatcher = new faceapi.FaceMatcher(labeledDescriptors);

  if (!faceMatcher) {
    console.error("Failed to load face matcher.");
    return;
  }
  
  console.log("Face Matcher loaded with descriptors:", faceMatcher);

  imageUpload.addEventListener('change', async () => {
    if (image) image.remove();
    if (canvas) canvas.remove();

    const file = imageUpload.files[0];
    if (!file) return;

    // Create an image element
    image = new Image();
    image.src = URL.createObjectURL(file);

    // Wait until the image is fully loaded
    image.onload = async () => {
      document.body.append(image);

      // Ensure the canvas is positioned over the image and absolutely positioned
      canvas = faceapi.createCanvasFromMedia(image);
      canvas.style.position = 'absolute';
      canvas.style.top = `${image.offsetTop}px`;
      canvas.style.left = `${image.offsetLeft}px`;
      canvas.width = image.width;
      canvas.height = image.height;
      document.body.append(canvas);

      const context = canvas.getContext('2d', { willReadFrequently: true });

      // Debugging: log image and canvas dimensions
      console.log(`Image dimensions: ${image.width}x${image.height}`);
      console.log(`Canvas dimensions: ${canvas.width}x${canvas.height}`);

      const displaySize = { width: image.width, height: image.height };
      faceapi.matchDimensions(canvas, displaySize);

      try {
        // Detect faces and generate descriptors
        const detections = await faceapi.detectAllFaces(image).withFaceLandmarks().withFaceDescriptors();
        console.log("Detections:", detections);  // Log detections for debugging

        // Resize the detections to match the display size of the image
        const resizedDetections = faceapi.resizeResults(detections, displaySize);

        // Clear the canvas before drawing
        context.clearRect(0, 0, canvas.width, canvas.height);

        // Draw bounding boxes
        faceapi.draw.drawDetections(canvas, resizedDetections);

        // Log the bounding box values for debugging
        resizedDetections.forEach((detection, i) => {
          console.log(`Bounding Box for face ${i}:`, detection.detection.box);
        });

        // Match detected faces with labeled faces using faceMatcher
        const results = resizedDetections.map(d => {
          return faceMatcher.findBestMatch(d.descriptor);
        }).filter(result => result !== null);  // Filter out any null results

        // Draw recognized face labels on the canvas
        results.forEach((result, i) => {
          const box = resizedDetections[i].detection.box;
          const drawBox = new faceapi.draw.DrawBox(box, { label: result.toString() });
          drawBox.draw(canvas);
        });

        // Display recognized names in the 'recognizedNames' div
        const recognizedNamesDiv = document.getElementById('recognizedNames');
        if (results.length > 0) {
          const recognizedNames = results.map(result => result.toString()).join(', ');
          recognizedNamesDiv.innerText = `Recognized Faces: ${recognizedNames}`;
        } else {
          recognizedNamesDiv.innerText = "No faces recognized.";
        }

        if (detections.length === 0) {
          console.log("No faces detected.");
        } else {
          console.log(`${detections.length} face(s) detected.`);
        }
      } catch (error) {
        console.error("Error during face detection:", error);
      }
    };
  });
});

// Load labeled images for face matching
async function loadLabeledImages() {
  const labels = ['Black Widow', 'Captain America', 'Captain Marvel', 'Hawkeye', 'Jim Rhodes', 'Thor', 'Tony Stark'];
  const labeledFaceDescriptors = await Promise.all(
    labels.map(async label => {
      const descriptions = [];
      for (let i = 1; i <= 2; i++) {
        try {
          const img = await faceapi.fetchImage(`https://raw.githubusercontent.com/WebDevSimplified/Face-Recognition-JavaScript/master/labeled_images/${label}/${i}.jpg`);
          const detections = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor();
          if (detections) {
            descriptions.push(detections.descriptor);
          } else {
            console.warn(`No face detected for ${label} image ${i}`);
          }
        } catch (error) {
          console.error(`Error fetching image for ${label}:`, error);
        }
      }
      return descriptions.length > 0 ? new faceapi.LabeledFaceDescriptors(label, descriptions) : null;
    })
  );

  return labeledFaceDescriptors.filter(desc => desc !== null);
}

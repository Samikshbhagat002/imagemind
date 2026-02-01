import { GoogleGenerativeAI, HarmBlockThreshold, HarmCategory } from "@google/generative-ai";
import Base64 from 'base64-js';
import MarkdownIt from 'markdown-it';
import { maybeShowApiKeyBanner } from './gemini-api-banner';
import './style.css';

let API_KEY = 'AIzaSyBah1J4Ut1xLBZtEfGRX_LDuxMaDqnBSgE';

let form = document.querySelector('form');
let promptInput = document.querySelector('input[name="prompt"]');
let output = document.querySelector('.output');
let customImageUpload = document.getElementById('custom-image-upload');
let imagePreview = document.getElementById('image-preview');
let imageChoices = document.querySelectorAll('input[name="chosen-image"]');
let uploadButton = document.getElementById('upload-button');

// Function to update the image preview
const updateImagePreview = () => {
  const selectedValue = form.elements['chosen-image'].value;

  if (selectedValue === 'custom') {
    if (customImageUpload.files && customImageUpload.files[0]) {
      const reader = new FileReader();
      reader.onload = function(e) {
        imagePreview.src = e.target.result;
      };
      reader.readAsDataURL(customImageUpload.files[0]);
    } else {
      imagePreview.src = ''; // Clear the preview if no file is selected
    }
  } else {
    imagePreview.src = selectedValue;
  }
};

// Event listeners for radio buttons
imageChoices.forEach(radio => {
  radio.addEventListener('change', () => {
    updateImagePreview();
  });
});

// Event listener for custom image upload
customImageUpload.addEventListener('change', () => {
  if (form.elements['chosen-image'].value === 'custom') {
    updateImagePreview();
  }
});

// Event listener for the upload button to trigger file input
uploadButton.addEventListener('click', () => {
  customImageUpload.click();
});

// Handle form submission
form.onsubmit = async (ev) => {
  ev.preventDefault();
  output.textContent = 'Generating...';

  try {
    let imageBase64;

    // Check if a custom image is uploaded
    if (form.elements['chosen-image'].value === 'custom' && customImageUpload.files.length > 0) {
      let file = customImageUpload.files[0];
      imageBase64 = await new Promise((resolve, reject) => {
        let reader = new FileReader();
        reader.onloadend = () => resolve(reader.result.split(',')[1]);
        reader.onerror = reject;
        reader.readAsDataURL(file);
      });
    } else {
      // Load the image as a base64 string for preselected images
      let imageUrl = form.elements['chosen-image'].value;
      imageBase64 = await fetch(imageUrl)
        .then(r => r.arrayBuffer())
        .then(a => Base64.fromByteArray(new Uint8Array(a)));
    }

    // Assemble the prompt by combining the text with the chosen or uploaded image
    let contents = [
      {
        role: 'user',
        parts: [
          { inline_data: { mime_type: 'image/jpeg', data: imageBase64, } },
          { text: promptInput.value }
        ]
      }
    ];

    // Call the multimodal model, and get a stream of results
    const genAI = new GoogleGenerativeAI(API_KEY);
    const model = genAI.getGenerativeModel({
      model: "gemini-1.5-flash", // or gemini-1.5-pro
      safetySettings: [
        {
          category: HarmCategory.HARM_CATEGORY_HARASSMENT,
          threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        },
      ],
    });

    const result = await model.generateContentStream({ contents });

    // Read from the stream and interpret the output as markdown
    let buffer = [];
    let md = new MarkdownIt();
    for await (let response of result.stream) {
      buffer.push(response.text());
      output.innerHTML = md.render(buffer.join(''));
    }
  } catch (e) {
    output.innerHTML += '<hr>' + e;
  }
};

// You can delete this once you've filled out an API key
maybeShowApiKeyBanner(API_KEY);

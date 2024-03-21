import os
import base64
from flask import Flask, request, jsonify
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

app = Flask(__name__)

# Set the values of your computer vision endpoint and computer vision key
# as environment variables:
try:
    endpoint = os.environ["VISION_ENDPOINT"]
    key = os.environ["VISION_KEY"]
except KeyError:
    print("Missing environment variable 'VISION_ENDPOINT' or 'VISION_KEY'")
    print("Set them before running this sample.")
    exit()

# Create an Image Analysis client
client = ImageAnalysisClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(key)
)

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    # Check if a base64 image is provided in the request
    if 'image' not in request.json:
        return jsonify({'error': 'No image provided'}), 400

    image_base64 = request.json['image']

    # Decode the base64 image
    image_data = base64.b64decode(image_base64)

    # Extract text (OCR) from an image stream. This will be a synchronously (blocking) call.
    result = client.analyze(
        image_data=image_data,
        visual_features=[VisualFeatures.READ]
    )

    # Process OCR results
    ocr_results = {'text_lines': []}
    if result.read is not None:
        for line in result.read.blocks[0].lines:
            line_data = {
                'text': line.text,
                'bounding_box': [(point.x, point.y) for point in line.bounding_polygon],
                'words': [{'text': word.text, 'bounding_polygon': [(point.x, point.y) for point in word.bounding_polygon], 'confidence': word.confidence} for word in line.words]
            }
            ocr_results['text_lines'].append(line_data)

    # Return OCR results
    response = {
        'image_height': result.metadata.height,
        'image_width': result.metadata.width,
        'model_version': result.model_version,
        'ocr_results': ocr_results
    }
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)

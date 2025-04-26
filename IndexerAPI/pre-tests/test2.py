from PIL import Image
import io
import base64

# Create a blank white image
image = Image.new("RGB", (128, 128), color="white")

# Convert PIL Image to bytes
buffer = io.BytesIO()
image.save(buffer, format="PNG")
image_bytes = buffer.getvalue()

# Encode to base64
base64_encoded = base64.b64encode(image_bytes).decode('utf-8')

# Print the base64 string
print(base64_encoded)
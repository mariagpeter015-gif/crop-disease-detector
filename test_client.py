import requests
import os

# =========================
# API URL
# =========================
url = "http://localhost:8000/predict"

# =========================
# Test Images
# =========================
test_images = [
    r"C:\Users\maria\OneDrive\Documents\crop-disease-detector\hk.JPG",
    r"C:\Users\maria\OneDrive\Documents\crop-disease-detector\leafeb.jpg",
    r"C:\Users\maria\OneDrive\Documents\crop-disease-detector\lbleaf.jpg"
]

# =========================
# Send Requests
# =========================
for image_path in test_images:

    try:

        with open(image_path, "rb") as image:

            response = requests.post(
                url,
                files={"file": image}
            )

        print("\n" + "=" * 50)

        print("Image:", os.path.basename(image_path))

        print("Status Code:", response.status_code)

        if response.status_code == 200:

            result = response.json()

            print("Prediction:", result["prediction"])

            print("Confidence:", result["confidence"], "%")

        else:

            print("Error Response:")

            print(response.text)

    except FileNotFoundError:

        print(f"\nImage not found: {image_path}")

    except Exception as e:

        print(f"\nError processing {image_path}")

        print("Reason:", str(e))

print("\nIntegration Testing Completed!")
import httpx

def test_hf():
    # Test feature extraction endpoint
    url = "https://api-inference.huggingface.co/pipeline/feature-extraction/intfloat/multilingual-e5-small"
    payload = {"inputs": ["query: What is diabetes?", "passage: Diabetes is a disease."]}
    
    try:
        response = httpx.post(url, json=payload, timeout=10.0)
        print("Status code:", response.status_code)
        if response.status_code == 200:
            res_data = response.json()
            print("Embedding list length:", len(res_data))
            print("Dimension of first embedding:", len(res_data[0]))
        else:
            print("Error response:", response.text)
    except Exception as e:
        print("Request failed:", e)

if __name__ == "__main__":
    test_hf()

# Vertex AI Custom Container Training - Complete Guide

# 📋 Overview

https://claude.ai/public/artifacts/8beb1199-b258-4bd1-9db0-9a6c77dc1afd

![image.png](image.png)

**What You'll Learn:**

- Train ML models using custom Docker containers on Vertex AI
- Deploy models to production endpoints
- Make predictions via UI and REST API

**Why Custom Containers?**

- Full control over dependencies and libraries
- Use any ML framework (scikit-learn, TensorFlow, PyTorch, XGBoost)
- Reproducible training environment

---

## 🏗️ Architecture Flow

```
Dataset (GCS) → Custom Container → Training Job → Model Artifact (GCS)
→ Model Registry → Endpoint → Predictions

```

---

## 📚 Step-by-Step Implementation

### **1. Setup Cloud Storage**

**Purpose:** Store training data and model artifacts

```bash
# Bucket Configuration
Name: custom_container_training
Region: us-central1 (Iowa)
Storage Class: Standard
Access Control: Uniform
Public Access: Prevented

```

**Actions:**

1. Open Cloud Storage  [https://console.cloud.google.com/storage/browser](https://console.cloud.google.com/storage/browser)
2. Create Bucket
3. Upload your dataset (e.g., `IRIS.csv`)
    
    **`gsutil_URI =** gs://my_custom_container_01` 
    
    ![image.png](image%201.png)
    

---

### **2. Create Vertex AI Notebook**

**Purpose:** Development environment for code and testing

**Access:** Vertex AI → Workbench → Managed Notebooks → Create

[https://console.cloud.google.com/vertex-ai/workbench/managed](https://console.cloud.google.com/vertex-ai/workbench/instances)

**Configuration:**

```
Name: custom-training-notebook
Region: us-central1
Machine: 4 vCPUs, 15 GB RAM, No GPU
Disk: SSD Persistent Disk
Service Account: Compute Engine default

Optional Settings:
- Idle shutdown: 15 minutes
- Enable file download, terminal, nbconvert

```

---

### **3. Write Training Code**

**File:** `training.ipynb` (later converted to `train.py`)

### **Core Training Logic**

```python
# ========== IMPORTS ==========
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle
import logging
from google.cloud import storage

# Configure logging
logging.basicConfig(level=logging.INFO)

# ========== LOAD DATA FROM GCS ==========
# GCS paths work directly with pandas
data = pd.read_csv("gs://custom_container_training/IRIS.csv")

# ========== PREPARE FEATURES ==========
array = data.values
X = array[:, 0:4]  # First 4 columns: sepal/petal measurements
y = array[:, 4]     # Last column: species label

# ========== TRAIN/TEST SPLIT ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,    # 80/20 split
    random_state=1     # Reproducible results
)

# ========== TRAIN MODEL ==========
svn = SVC()  # Support Vector Classifier
svn.fit(X_train, y_train)

# ========== EVALUATE ==========
predictions = svn.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
logging.info(f"Model Accuracy: {accuracy:.4f}")  # ~0.9666 (96.66%)

# ========== SERIALIZE MODEL ==========
model_filename = "model.pkl"
with open(model_filename, "wb") as f:
    pickle.dump(svn, f)

# ========== UPLOAD TO GCS ==========
bucket_name = "custom_container_training"
storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)
blob = bucket.blob("model.pkl")
blob.upload_from_filename(model_filename)

logging.info(f"Model exported to gs://{bucket_name}/model.pkl")

```

**Key Concepts:**

- **GCS Integration:** Use `gs://` paths directly with pandas
- **Serialization:** `pickle.dump()` saves the trained model
- **Cloud Storage Client:** Uploads artifacts to GCS

---

### **4. Create Docker Container**

**Purpose:** Package training code with all dependencies

### **Step 4.1: Convert Notebook to Script**

```bash
# In JupyterLab Terminal
cd custom-container
jupyter nbconvert training.ipynb --to python
mv training.py train.py  # Rename for clarity

```

### **Step 4.2: Define Dependencies**

**File:** `requirements.txt`

```
pandas
scikit-learn==0.24.0
fsspec
gcsfs
google-cloud-storage

```

**Why These Libraries?**

- `pandas`: Data manipulation
- `scikit-learn==0.24.0`: ML algorithms (version pinned for reproducibility)
- `fsspec` + `gcsfs`: Enable `gs://` path reading
- `google-cloud-storage`: Upload artifacts to GCS

### **Step 4.3: Create Dockerfile**

**File:** `Dockerfile`

```docker
# Base image with Python 3.7
FROM python:3.7-buster

# Set working directory
WORKDIR /root

# Copy files into container
COPY train.py /root/train.py
COPY requirements.txt /root/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run training script when container starts
ENTRYPOINT ["python", "train.py"]

```

**Dockerfile Explained:**

1. **FROM:** Start with official Python 3.7 image
2. **WORKDIR:** All commands run from `/root`
3. **COPY:** Add training files to container
4. **RUN:** Install Python packages at build time
5. **ENTRYPOINT:** Execute `train.py` when container launches

---

### **5. Build and Push Docker Image**

### **Step 5.1: Create Artifact Registry Repository**

[https://console.cloud.google.com/artifacts](https://console.cloud.google.com/artifacts)

```bash
# Create repository (one-time setup)
gcloud artifacts repositories create iris-custom-repo \
    --repository-format=docker \
    --location=us-central1 \
    --description="Custom training containers"

```

### **Step 5.2: Build Image**

```bash
# Build Docker image
docker build -t iris_custom:v1 .

# Tag for Artifact Registry
docker tag iris_custom:v1 \
    us-central1-docker.pkg.dev/YOUR_PROJECT_ID/iris-custom-repo/iris_custom:v1

```

### **Step 5.3: Push to Registry**

```bash
# Authenticate Docker
gcloud auth configure-docker us-central1-docker.pkg.dev

# Push image
docker push us-central1-docker.pkg.dev/YOUR_PROJECT_ID/iris-custom-repo/iris_custom:v1

```

---

### **6. Run Training Job on Vertex AI**

### **Console Steps:**

1. **Navigate:** Vertex AI → Training → Create
2. **Dataset:** Select "No managed dataset"
3. **Training Method:** Custom training

**Model Details:**

```
Model Name: iris_custom_training
Service Account: Compute Engine default

```

**Container Settings:**

```
Container Type: Custom container
Image: us-central1-docker.pkg.dev/YOUR_PROJECT_ID/iris-custom-repo/iris_custom:v1

```

**Compute Resources:**

```
Machine Type: n1-standard-4 (4 vCPU, 15 GB RAM)
Disk: 100 GB SSD

```

**Other Settings:**

- Hyperparameter Tuning: Disabled
- Deployment: Skip (deploy manually later)
1. **Start Training** → Job takes ~5-10 minutes

**Verify Success:**

- Training pipeline shows ✅ green check
- `model.pkl` appears in GCS bucket

---

### **7. Import Model to Registry**

**Purpose:** Version control and deployment readiness

### **Console Steps:**

1. **Navigate:** Vertex AI → Model Registry → Import
2. **Configure:**

```
Model Name: iris_custom_model
Model Type: Prebuilt container
Framework: scikit-learn
Framework Version: 0.24
Artifact Location: gs://custom_container_training/model.pkl
```

1. **Import** → Completes in ~1 minute

---

### **8. Deploy Model to Endpoint**

**Purpose:** Create online prediction service

### **Deployment Settings:**

```
Endpoint Name: iris_endpoint
Traffic Split: 100% to this model version

Resources:
- Min Nodes: 1
- Max Nodes: 1
- Machine Type: n1-standard-2 (2 vCPU, 7.5 GB RAM)

Service Account: Compute Engine default
Encryption: Google-managed
Logging: Enabled (request/response)

```

**Deploy** → Takes ~5-10 minutes

---

## 🧪 Testing Predictions

### **Method 1: Vertex AI UI**

1. Go to Model Registry → `iris_custom_model` → Deploy & Test
2. Input test data:

![image.png](image%202.png)

```json
{
  "instances": [
    [5.1, 3.5, 1.4, 0.2],
    [7.9, 3.8, 6.4, 2.0]
  ]
}

```

**Expected Output:**

```json
{
  "predictions": ["setosa", "virginica"]
}

```

---

### **Method 2: REST API (cURL)**

### **Setup Authentication:**

```bash
# Authenticate (one-time)
gcloud auth application-default login

# Set environment variables
export PROJECT_ID="YOUR_PROJECT_ID"
export ENDPOINT_ID="YOUR_ENDPOINT_ID"
export REGION="us-central1"

```

### **Create Input File:**

**File:** `input.json`

```json
{
  "instances": [
    [5.1, 3.5, 1.4, 0.2],
    [7.9, 3.8, 6.4, 2.0]
  ]
}

```

### **Make Prediction Request:**

```bash
curl \
  -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $(gcloud auth application-default print-access-token)" \
  "https://${REGION}-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${REGION}/endpoints/${ENDPOINT_ID}:predict" \
  -d @input.json

```

**Response:**

```json
{"predictions": ["setosa", "virginica"]}

```

---

### **Method 3: Python SDK**

```python
from google.cloud import aiplatform

# Initialize endpoint
endpoint = aiplatform.Endpoint(
    "projects/YOUR_PROJECT_ID/locations/us-central1/endpoints/YOUR_ENDPOINT_ID"
)

# Prepare test data
instances = [
    [5.1, 3.5, 1.4, 0.2],  # Setosa features
    [7.9, 3.8, 6.4, 2.0]   # Virginica features
]

# Get predictions
response = endpoint.predict(instances=instances)
print(response.predictions)  # ['setosa', 'virginica']

```

---

## 📊 Complete Workflow Summary

| Step | Action | Output |
| --- | --- | --- |
| 1 | Create GCS bucket | Storage ready |
| 2 | Upload dataset | `IRIS.csv` in GCS |
| 3 | Write training code | `train.py` |
| 4 | Create Dockerfile | Container definition |
| 5 | Build & push image | Image in Artifact Registry |
| 6 | Run training job | `model.pkl` in GCS |
| 7 | Import to registry | Versioned model |
| 8 | Deploy endpoint | Online predictions ready |
| 9 | Test predictions | ✅ Working! |

---

## 🎯 Key Takeaways

### **Custom vs Prebuilt Containers**

| Aspect | Custom Container | Prebuilt Container |
| --- | --- | --- |
| Flexibility | Full control | Limited to Google frameworks |
| Setup | Requires Dockerfile | No Docker needed |
| Dependencies | Any library | Predefined versions |
| Use Case | Complex/custom workflows | Standard ML frameworks |

### **Best Practices**

✅ **DO:**

- Pin dependency versions (`scikit-learn==0.24.0`)
- Use Artifact Registry (not deprecated `gcr.io`)
- Enable request/response logging
- Set idle shutdown for notebooks

❌ **DON'T:**

- Hardcode credentials in code
- Use outdated Python versions
- Skip error handling in training scripts
- Forget to clean up resources

---

## 🔗 Quick Reference

**GCS Path Format:**

```
gs://BUCKET_NAME/path/to/file

```

**Artifact Registry Image Format:**

```
REGION-docker.pkg.dev/PROJECT_ID/REPO_NAME/IMAGE_NAME:TAG

```

**Endpoint URL Format:**

```
https://REGION-aiplatform.googleapis.com/v1/projects/PROJECT_ID/locations/REGION/endpoints/ENDPOINT_ID:predict

```

---

## 🚀 Next Steps

1. **Batch Predictions:** Process large datasets asynchronously
2. **Model Monitoring:** Track prediction drift and performance
3. **CI/CD Integration:** Automate training/deployment pipelines
4. **Hyperparameter Tuning:** Optimize model performance

---

## 💡 Mnemonics

**TRAINING** - Remember the workflow:

- **T**ask definition (what to train)
- **R**esources setup (GCS, notebooks)
- **A**rtifacts creation (code, Dockerfile)
- **I**mage building (Docker)
- **N**ode allocation (compute resources)
- **I**nitiate job (start training)
- **N**etwork endpoint (deploy)
- **G**et predictions (test)

**4D Container Checklist:**

1. **D**ependencies (`requirements.txt`)
2. **D**ockerfile (build instructions)
3. **D**eploy (push to registry)
4. **D**eliver (run training job)
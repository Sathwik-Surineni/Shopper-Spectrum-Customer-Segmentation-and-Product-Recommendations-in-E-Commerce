# 🛒 Shopper Spectrum: Customer Segmentation & Product Recommendations in E-Commerce

Welcome to the **Shopper Spectrum** project!  
This project explores transactional data from an online retail store to understand customer behavior, segment customers using RFM (Recency, Frequency, Monetary) analysis, and recommend similar products using collaborative filtering.

---

##  Features

-  **Customer Segmentation** using KMeans clustering
-  **Product Recommendations** using cosine similarity
-  **Machine Learning Models** generated from raw transaction data
-  Simple and interactive **Streamlit UI**

---

##  Project Structure

```bash
shopperspectrum/
│
├── app.py                  # Streamlit app to run the UI
├── generate_models.py      # Script to train & generate model files (.pkl)
├── main.py                 # Data cleaning, transformation & RFM logic
├── data.csv                # Input dataset
├── scaler.pkl              # StandardScaler model (generated)
├── kmeans.pkl              # KMeans clustering model (generated)
├── product_similarity.pkl  # Product similarity matrix (generated)
└── README.md               # Project documentation
```
## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Sathwik-Surineni/Shopper-Spectrum-Customer-Segmentation-and-Product-Recommendations-in-E-Commerce.git
cd shopperspectrum
```
### 2. Create & Activate a Virtual Environment
```bash
python -m venv .venv
source .venv/Scripts/activate     # For Git Bash or PowerShell on Windows
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
 If requirements.txt is missing, install manually:

pip install pandas numpy scikit-learn joblib streamlit
```
### 4. Generate Model Files (Optional if already present)
```bash
python generate_models.py
```
### Run the App
```bash
streamlit run app.py
```
Then open the link in your browser (e.g., http://localhost:8501).
## How It Works
Customer Segmentation: Based on Recency, Frequency, and Monetary (RFM) values.

Product Recommendation: Based on item-item similarity using cosine distance.

## Use Cases
E-commerce dashboards

Marketing segmentation

Personalized recommendation systems

RFM-based business insights





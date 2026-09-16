# 📊 Customer Intelligence Dashboard

> An interactive business intelligence dashboard built with **Python, Streamlit, Pandas, NumPy, and Plotly** to analyze customer behavior, segmentation, revenue, and churn risk.

## 🚀 Live Demo

**Live Dashboard:**
[Customer Intelligence Dashboard](https://customerdashboard-iejuws2ipk46pqkefelix3.streamlit.app/)

---

## 🎯 Project Overview

Businesses generate large volumes of customer data, but raw data alone does not provide actionable insight.

The **Customer Intelligence Dashboard** transforms customer-level data into an interactive analytics system that helps users understand:

* Customer distribution across segments
* Revenue contribution
* Customer churn
* Churn-risk patterns
* High-risk customers requiring attention
* Customer behavior based on engagement and support activity

The application provides an easy-to-use interface where users can filter existing data or upload their own compatible CSV dataset.

---

## 💡 Key Features

### 📌 KPI Monitoring

The dashboard provides high-level business metrics including:

* 👥 Total Customers
* 💰 Monthly Revenue
* 📉 Churn Rate
* ⚠️ High-Risk Customers

### 👥 Customer Segmentation

Analyze customers across different segments such as:

* Premium
* Standard
* Value

The segmentation view helps identify the size and revenue contribution of different customer groups.

### 📉 Churn Analytics

The dashboard provides:

* Overall churn rate
* Churn rate by customer segment
* Churn-risk distribution
* High-risk customer identification

### 🔎 Interactive Filtering

Users can dynamically filter the dashboard based on:

* Customer segment
* Contract type
* Churn-risk threshold

All visualizations and KPIs update based on the selected filters.

### 📂 CSV Upload

Users can upload their own customer dataset through the Streamlit interface.

This makes the dashboard reusable beyond the included demonstration dataset.

### ⚠️ High-Risk Customer Watchlist

The dashboard generates a prioritized view of customers with higher estimated churn risk, including relevant customer attributes such as:

* Customer ID
* Segment
* Contract
* Revenue
* Tenure
* Support tickets
* Monthly logins
* Churn risk

---

## 🏗️ System Architecture

```text
                    ┌─────────────────────┐
                    │   Customer Data     │
                    │ CSV / File Upload   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │       Pandas        │
                    │ Data Processing     │
                    │ Validation/Filter   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Analytics Engine    │
                    │                     │
                    │ • KPI Calculation   │
                    │ • Segmentation      │
                    │ • Churn Analysis    │
                    │ • Risk Analysis     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      Plotly         │
                    │ Interactive Charts  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     Streamlit       │
                    │ Interactive Dashboard│
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      End User       │
                    │ Business Insights   │
                    └─────────────────────┘
```

---

## ☁️ Cloud Deployment

The application has been deployed to **Streamlit Community Cloud**, making the dashboard accessible through a web browser without requiring local installation.

### Deployment Workflow

```text
GitHub Repository
       │
       ▼
Streamlit Community Cloud
       │
       ▼
Install Dependencies
       │
       ▼
Run app.py
       │
       ▼
Public Dashboard
```

### Deployment Configuration

| Configuration   | Value                           |
| --------------- | ------------------------------- |
| Application     | Customer Intelligence Dashboard |
| Framework       | Streamlit                       |
| Main File       | `app.py`                        |
| Deployment      | Streamlit Community Cloud       |
| Dependency File | `requirements.txt`              |
| Data Format     | CSV                             |
| Visualization   | Plotly                          |

---

## 🛠️ Technology Stack

### Programming & Data

* **Python**
* **Pandas**
* **NumPy**

### Visualization

* **Plotly**

### Application Framework

* **Streamlit**

### Deployment

* **Streamlit Community Cloud**

### Version Control

* **GitHub**

---

## 📁 Project Structure

```text
customer-intelligence-streamlit/
│
├── app.py
├── requirements.txt
├── README.md
├── ARCHITECTURE.md
├── GITHUB_SETUP.md
├── LINKEDIN_REFLECTION.md
│
├── data/
│   └── customer_data.csv
│
└── screenshots/
    └── dashboard_mockup.svg
```

---

## ⚙️ Run Locally

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd customer-intelligence-streamlit
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the dashboard

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📊 Dataset

The repository contains a **synthetic customer dataset** created for demonstration and learning purposes.

Example attributes include:

| Feature        | Description                    |
| -------------- | ------------------------------ |
| CustomerID     | Unique customer identifier     |
| Age            | Customer age                   |
| TenureMonths   | Customer tenure                |
| MonthlyRevenue | Monthly customer revenue       |
| SupportTickets | Number of support interactions |
| MonthlyLogins  | Monthly platform logins        |
| Contract       | Customer contract type         |
| Segment        | Customer segment               |
| ChurnRisk      | Estimated churn-risk score     |
| Churn          | Churn indicator                |

> **Note:** The included dataset is synthetic and should not be interpreted as real customer or production business data.

---

## 📈 Business Questions Addressed

The dashboard is designed around practical business questions:

**1. How large is the customer base?**

Monitor the number of active records under the selected filters.

**2. How much revenue is represented?**

Track aggregate monthly revenue across customer groups.

**3. Which customer segments have higher churn?**

Compare churn rates across different segments.

**4. Which customers may require attention?**

Identify customers with higher estimated churn risk.

**5. Can the dashboard analyze another dataset?**

Yes. Users can upload a compatible CSV through the dashboard.

---

## 🔐 Data & Privacy

The demonstration dataset does not contain real personally identifiable customer information.

For production usage, appropriate security controls should be implemented for:

* Personally identifiable information
* Authentication
* Authorization
* Data encryption
* Secure data storage
* Secrets and API credentials
* Audit logging

---

## ⚠️ Current Limitations

This project is a learning/prototype analytics system.

Current limitations include:

* The included customer dataset is synthetic.
* Churn-risk values in the demonstration dataset are generated using a simplified analytical approach.
* The dashboard is not connected to a production CRM or data warehouse.
* No authentication or role-based access control is implemented.
* Real-time streaming data is not currently connected.

---

## 🔮 Future Enhancements

Potential production improvements include:

* 🤖 Integrate a validated ML churn-prediction model
* 🔄 Connect to live databases or APIs
* 📡 Add real-time customer activity monitoring
* 🔐 Add authentication and role-based access
* 📊 Add cohort and retention analysis
* 💰 Add customer lifetime value prediction
* 🚨 Add automated churn alerts
* 📈 Add time-series revenue forecasting
* ☁️ Introduce a production data pipeline
* 🧠 Add explainable AI for churn predictions

---

## 📚 Day 44 & Day 45 Learning Journey

### Day 44 — Interactive Analytics Systems

Built the interactive Customer Intelligence Dashboard and focused on:

* Business intelligence
* Customer analytics
* Interactive visualization
* Streamlit application development
* Churn analysis
* Dashboard design

### Day 45 — Cloud Deployment

Deployed the dashboard to the cloud and focused on:

* Cloud deployment
* Dependency configuration
* Public application accessibility
* Deployment testing
* Production-oriented documentation

---

## 🎓 Key Learning

The main lesson from this project was that **analytics becomes more useful when it is converted into an interactive decision-support system**.

Instead of presenting users with static analysis, the dashboard allows them to explore customer segments, revenue, churn, and risk dynamically.

The project also provided practical experience in moving from:

```text
Raw Data
   ↓
Data Analysis
   ↓
Business Insights
   ↓
Interactive Dashboard
   ↓
Cloud Deployment
```

---

## 🚀 Future Vision

The long-term goal is to evolve this prototype into a complete **Customer Intelligence Platform** capable of combining:

**Customer Data + Machine Learning + Real-Time Analytics + Business Intelligence**

to help organizations identify customer-risk patterns and make data-driven retention decisions.

---

## 👨‍💻 Project Status

**Status:** ✅ Deployed

**Day:** 45 — Cloud Deployment

**Application:** Customer Intelligence Dashboard

**Live Demo:**
[Open Dashboard](https://customerdashboard-iejuws2ipk46pqkefelix3.streamlit.app/)

---

## 📌 Author

Built as part of a structured **Data Science & Analytics learning journey**, progressing from data preparation and machine learning to interactive analytics and cloud deployment.

---

⭐ If you find this project useful, consider exploring the repository and sharing your feedback.

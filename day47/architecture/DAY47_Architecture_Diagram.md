# CustomerIQ Architecture Diagram

```text
                         CUSTOMER INTELLIGENCE PLATFORM
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │       DATA SOURCES      │
                         │ Orders • Customers      │
                         │ Products • Payments     │
                         │ Reviews • Delivery      │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │   DATA INGESTION & EDA   │
                         │ Loading • Profiling      │
                         │ Missing Values • Quality │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │ DATA PREPROCESSING       │
                         │ Cleaning • Joins         │
                         │ Transformation           │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │ FEATURE ENGINEERING      │
                         │ RFM • Behavioral • Time  │
                         │ Transaction Features    │
                         └────────────┬────────────┘
                                      │
                ┌─────────────────────┼─────────────────────┐
                ▼                     ▼                     ▼
       ┌────────────────┐    ┌────────────────┐    ┌────────────────┐
       │ SEGMENTATION   │    │ CHURN MODEL    │    │ FORECASTING    │
       │ RFM + K-Means  │    │ Classification │    │ Time Series    │
       └───────┬────────┘    └───────┬────────┘    └───────┬────────┘
               │                     │                     │
               └─────────────────────┼─────────────────────┘
                                     ▼
                         ┌─────────────────────────┐
                         │   ANALYTICS ENGINE      │
                         │ KPIs • Model Metrics    │
                         │ Insights • Recommendations│
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │   CUSTOMERIQ DASHBOARD  │
                         │ KPIs • Segments          │
                         │ Churn • Value • Forecast │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │     BUSINESS ACTIONS    │
                         │ Retention • Targeting   │
                         │ Revenue Planning         │
                         └─────────────────────────┘
```

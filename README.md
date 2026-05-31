Credit Scoring Business Understanding
1. The Basel II Accord requires financial institutions to measure, monitor, and manage credit risk using transparent and reliable methodologies. Since regulatory capital requirements may depend on model outputs, regulators expect institutions to understand and justify how risk estimates are generated.Under Basel II, a highly accurate model is not sufficient by itself. The model must also be explainable, reproducible, and defensible to regulators, auditors, and internal risk committees.

2. Supervised machine learning models require a target variable (label) to learn patterns from historical data. If a dataset does not contain a direct indicator such as: 
    Defaulted loan
    Bankruptcy event
    Charge-off status
then the organization must construct a proxy variable that approximates credit risk.

Business Risks Introduced by Proxy Variables
    1. Label Misalignment: - The proxy may not accurately represent true default behavior. As a result, the model learns patterns associated with the proxy rather than actual default risk.

    2. Biased Risk Estimates: - If the proxy captures only part of the default process, risk predictions may be systematically biased.

    3. Regulatory Concerns: - Regulators may challenge whether the proxy appropriately represents the intended risk concept. Poorly justified proxies can weaken confidence in the model's outputs and governance framework.

    4. Business Decision Errors: - Inaccurate predictions can lead to:
        Rejecting profitable customers
        Approving high-risk customers
        Mispricing loans
        Incorrect capital allocation
    5. Model Drift: - The relationship between the proxy and actual default may change over time, reducing model effectiveness and requiring continuous monitoring.


3. key trade-offs between a simple, interpretable model (e.g., Logistic Regression with WoE) and a high-performance model (e.g., Gradient Boosting)

Logistic Regression with Weight of Evidence (WoE): - is often favored when transparency, explainability, and regulatory acceptance are critical.
    Advantages
        Easy to understand and explain
        Coefficients directly show risk impact
        Widely accepted in banking and credit scoring
        Simpler validation and governance
        Stable and transparent decision-making process
    Disadvantages
        Assumes relatively simple relationships
        Limited ability to capture nonlinear effects
        May underperform on complex datasets

Gradient Boosting: - may offer stronger predictive performance but requires additional controls, documentation, validation, and explainability techniques to satisfy regulatory expectations.
    Advantages
        Often delivers superior predictive accuracy
        Captures nonlinear relationships automatically
        Handles complex interactions between variables
        Can improve risk discrimination and ranking
    Disadvantages
        More difficult to explain
        Harder to validate and document
        Greater regulatory scrutiny
        Requires additional explainability methods
        Increased governance and monitoring burden
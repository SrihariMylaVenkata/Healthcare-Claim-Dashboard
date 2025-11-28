What I built

I put together a small project that predicts whether a healthcare claim is likely to be approved or denied. I cleaned a claims dataset, trained a simple baseline model, and then wrapped it in a Streamlit app so anyone can choose claim attributes and see the predicted denial probability with a clear verdict.

The data I used

I worked with a CSV that includes:

Outcome (Paid or Denied)

Numeric fields: Billed Amount, Allowed Amount, Paid Amount

Categorical fields: Procedure Code, Diagnosis Code, Insurance Type, Reason Code, Follow-up Required, AR Status

If a file had “Claim Status” instead of “Outcome,” I converted it to Outcome = Paid/Denied.

How I approached the analysis

Loaded the raw CSV and checked columns, types, and missing values.

Cleaned things up: parsed dates, standardized IDs as text (then dropped pure IDs/dates for modeling), stripped stray spaces/quotes, and capped extreme billed amounts to reduce outlier impact.

Created the label: Denied_Binary = 1 for Denied, 0 for Paid.

One-hot encoded categorical columns and scaled numeric ones.

Trained a simple logistic regression with class_weight="balanced" and measured accuracy/recall/AUC.

Explored quick visuals: denial rate by Insurance Type, Reason Code, and Follow-up Required.

Saved a cleaned CSV specifically for the app.

What the app does

Reads my cleaned CSV.

Trains the same preprocessing + model pipeline on the fly.

Builds the UI directly from the dataset, so dropdowns and options match the data.

Lets a user pick inputs and returns a denial probability with a “Likely Approved” or “Likely Denied” verdict.

Shows a short “why” panel that highlights inputs far from typical values (a quick hint, not a formal explanation).

How I run it locally (what I actually do)

Make sure Python is installed and available on PATH.

Install the required libraries once (Streamlit, pandas, scikit-learn, etc.).

Keep three things in one folder:

the Streamlit app file

the cleaned CSV

a requirements file (optional but handy)

Start Streamlit from that folder. It opens in my browser.

In the app, I either upload the CSV or point to the local path, select options, and hit Predict.

The user experience

A user uploads the dataset (or uses a local path), sees a quick summary (row count, denial rate, AUC), selects attributes like Insurance Type and Reason Code, and clicks Predict. The app shows a denial probability, a clear verdict, and a short hint panel to build intuition.

Deploying it

I pushed the app to GitHub and deployed on Streamlit Community Cloud. If deployment complains, it’s usually because the repo isn’t connected or the app file path isn’t set. Once the repo and entry file are correct, the app goes live.

Common snags I hit (and how I fixed them)

Missing modules: installed the package in the same Python environment I’m using to run Streamlit.

“File does not exist”: ran Streamlit from the folder that actually contains the app file.

Wrong label name: standardized on Outcome (Paid/Denied) and created Denied_Binary from that.

One-class data: made sure the sample had both Paid and Denied to train and evaluate meaningfully.

What this is (and isn’t)

This is a lightweight, educational tool to understand denial risk and play with inputs. It’s not a production claims engine. If I were to productionize it, I’d add stronger validation, versioning, monitoring, security, and strict handling of PHI/PII.

Why this matters

This setup lets me quickly show stakeholders how certain fields (like Reason Code, Follow-up Required, or Insurance Type) relate to denial risk, and it gives them a simple, interactive way to test “what if” scenarios without touching code.

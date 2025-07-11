# Fraud Detection Workshop Setup Instructions
The Following set of instructions guide you through the process of deploying and configuring a Domino environment for the "Fraud Detection" Workshop.  This set of documents assume you have a working knowledge of the platforms you are deploying and configuring.  They are not as explicit as user instructions (no screenshots)

Anywhere you see an upper case word in **BOLD** that is refering to a button that needs to be clicked.

## Setup Domino Environment
- [ ] Create Fleetcommand Domino Instance as Prescribed in the `./fleetcommand.md` file.
  
- [ ] Create a copy of checklist Template and work the checklist: `https://docs.google.com/spreadsheets/d/1fbP-eY0gCBw64YnrMXFIDbaLYcwlHrL65UgjarDWj4M/edit?gid=0#gid=0`
  
- [ ] Create Domino Compute Environment as per `./environment.md`

## Configure Data Source (Admin Section)

**CREATE DATA SOURCE**

- [ ] Select Data Source `Amazon S3`
- [ ] Bucket: `accesspoint-hbgerepzfgudec1u1f75k9r9qteswuse2a-s3alias`
- [ ] Region: `us-east-2`
- [ ] Data Source Name: `credit_card_fraud_detection`
- [ ] Data Source Description: `The dataset contains transactions made by credit cards in September 2013 by European cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.`

**NEXT**

- [ ] Credential Type: Select `Service Account`

**NEXT**

- [ ] If Nexus Data Planes: Select `Select All`

**NEXT**

- [ ] Access Key ID: Enter `<Your Access Key ID>`
- [ ] Secret Access Key: Enter`<Your Secret Access Key>`

**TEST CREDENTIALS**

- [ ] Update permissions by selecting the `Everyone` Radio Button

**FINISH SETUP**

## Other Admin Configurations

- [ ] Billing Tags: `Fraud Claims HR Marketing Operations Quant Underwriting`


## Create Donor Project

**Create Project**

- [ ] Template: `None`
- [ ] Project Name: `Fraud-Detection-Workshop-Donor`
- [ ] Visibility: `Public`

**NEXT**

- [ ] Hosted By: `Git Service Provider`
- [ ] Git Service Provider: `Github`
- [ ] Git Credentials: `None`
- [ ] Git Repo URL: `https://github.com/dominodatalab/Fraud-Detection-Workshop.git`

**CREATE**

- [ ] Set Default Compute Environment: `Fraud-Detection-Workshop`
- [ ] Add Data Source: `credit_card_fraud_detection`
- [ ] Add Tags `Fraud, Anomaly Detection, Python, XGBoost, AdaBoost (ADA) Gaussian Naive Bayes (GNB), API, App
`

## Create Template From Donor Project

**CREATE TEMPLATE**

- [ ] Template Name: `Fraud-Detection-Workshop-Template`
- [ ] Description:
- [ ] Access: `Anyone with access...`

**NEXT**

- [ ] Ensure `Select All` is selected and deselect the following:
  - Goals
  - Datasets
  - External Volumes
  - Artifacts
  - Imported Projects
  - Published Entities
  - Integrations
- [ ] Default Billing Tag: `Fraud`
- [ ] Default Environment: `Fraud-Detection-Workshop`
- [ ] Default Hardware Tier: `Small`

**NEXT**

- [ ] File Storage: `In new Repo...`
- [ ] Git Service Provider: `GitHub`
- [ ] Git Credentials: `<Admin Workshop Credentials>`
- [ ] Owner: `<Select Owner>`
- [ ] Repo Visibility: `Public`

**CREATE**








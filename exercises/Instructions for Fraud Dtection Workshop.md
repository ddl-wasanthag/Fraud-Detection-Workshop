Instructions for Fraud Dtection Workshop


# 1. Up and Running
In this phase, we will be preparing a project in which to build our model.  We will explore the QuickStart project, Examine a Project Template, Create a new Project from a Template, add Contributors to the project, and apply a Governance Policy

**New Domino Concepts**
- Home Page
- Projects
- Project Templates
- Policies & Governance

## Exercise Instructions

1. Open Domino Website (Explore)
    - You can always "Smash Domino" to go home
    - Explore Top Navigation Menu
    - Explore the Quickstart Project

2. Click "Develop" -> "Projects" (Top Menu Bar)

3. Select Templates

4. Click "Fraud Detection Workshop"
- and notice all the entries

5. Click "Create Project" button (lower right)

6. Configure the Project page as shown and click "Next"

7. Configure the Code page as shown and click "Create"

7. Explore your new Project

8. Click project "Settings"

9. Click "Access and Sharing"

10. Invite Collaborator under under "Collaborators and Permissions"

11. Set their role to "Contributor"

12. Add another user and set their role to "Results Consumer"

13.  Click "Govern" -> "Bundles" (Left-Hand Column)

14. Click "Create Bundle"

15. Configure Form as shown and click "Create"

16.  Provide a "Business Case" and click "Request Review"

17.  Once review has been approved, "transition Stage to "Stage 2"

THIS CONCLUDES THE "1. UP AND RUNNING" SECTION OF THE WORKSHOP







# Data Exploration
In this phase, we will begin the process of exploring a transactional dataset to see if it will be good for model training.  We will read a CSV file from a Domino Data Source into a Dataframe, clean the data, generate some visualizations, and save the data to a Domino Data Set.  We will begin from within your copy of the "Fraud Detection Workshop" projecy.

**New Domino Concepts**
- Workspaces
- Data Sources
- Datasets
- Compute Environments

## Exercise Instructions

1.  Click "Workspace" (Left-hand column) and click "Jupyter Lab"

2.  Click "Next" 4x and then "Create"
    - Feel free to look around but do not change anything.

3.  Review Workspace Creation Workflow

4.  Review Workspace UI & Lefthand Sidebar

5.  Open the Notebook /mnt/code..../.../...

6.  In Data Sources at Left, Copy the Python Snippet

7.  Paste Python Snippett in Cell Where Instructed

8.  Run notebook cells (either manually or automatically)

9.  Review generated Plots

10. Follow Path to Updated Location in Workspace.

11.  Save and Commit Code



# 3. Data Engineering
In this phase, we will continue to prepare the data for training by executing some simpleData Engineering tasks.  We will be execute a Domino job that reads the updated CSV from a Domino Data Set, performs simple feature engineering such as normalization the data, adding a derived column, saving the data, and taking a snapshot of the data.

**New Domino Concepts**
- Jobs
- Dataset Snapshots
- Artifacts

## Exercise Instructions

1.  Within the Workspace, open and review Engineering Python Script (do not modify)

2. with all code comitted, Click "Run Job"

3.  Fill out the form as shown Below, Go

4.  Back on the main tab, in the Project Click "Jobs" on the left.

5.  Expand the Right for details
    - Notice how everything updates itself in real time.

6.  Review the artifacts created by the job run.



# 4. Model Training & Evaluation
In this phase, we will simultaneously train 3 models, evaluate them, and register the best using a coordinated workflow.  We will execute a Domino Flow that trains the three models, evaluate the models using the Domino Experiment Manager, and register the best ones in the Model Registry.

**New Domino Concepts**
- Flows
- Experiment Manager
- Model Registry

## Exercise Instructions

In the workspace, open the terminal

Run the Flow

Watch the Flow Execution

Click "Experiment Manager"  (Main Window, Left-Hand Column)

Select all runs (3) and click "Compare"

Select the one with the best metrics (here is a hint: XGBoost)

Review the complete traceability and all factors.

Click "Register Model From Run" in Upper Right Hand

Create model name

In governnace 


# 5.  Delivery and hosting
In this phase, we will deploy the model for consumption by other users.  We will deploy the model as a REST API endpoint, and host a Streamlit app that calls the REST endpoint.

## Exercise Instructions


**New Domino Concepts**
- Model Endpoints
- Hosted Applications
 
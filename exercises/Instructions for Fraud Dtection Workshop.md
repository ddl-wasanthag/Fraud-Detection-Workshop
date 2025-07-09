Instructions for Fraud Dtection Workshop


# 1. Up and Running
In this phase, we will be preparing a project in which to build our model.  We will explore the QuickStart project, Examine a Project Template, Create a new Project from a Template, add Contributors to the project, and apply a Governance Policy

## New Domino Concepts
**Home Page:** 
When logging in to Domino, you will land on a homepage that gives you quick access to recent work, elevates timely tasks and notifications, and promotes data science assets recently published in your organization.

**Projects:**
Domino uses Projects to organize work for data science and analytics teams. Projects help teams run experiments and improve code. Using Projects, you can manage data, code, artifacts, and user permissions.

**Project Templates:**
Project templates are created from an existing project by selecting which assets to include (code, datasets, apps, etc.). With templates created, users can kickstart their projects from a collection of existing prototypes rather than beginning from scratch.

**Governance Policies & Bundles:**
Policies in Domino define the lifecycle of a scientific output, such as deploying a model to production, building a statistical analysis, or building an AI system.  A governed bundle can be a model, an application, a report, or any other asset developed within the context of a project. It will store all evidence related to the policy it governs and keep the lineage to the relevant attachments.



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

## New Domino Concepts
**Workspaces:** A Domino workspace is an interactive session where you can conduct research, analyze data, train models, and more. Use workspaces to work in the development environment of your choice, like Jupyter notebooks, RStudio, VS Code, and many other customizable environments.

**Data Source Connector:** 
Managed data connectors that can connect to SQL and file-type stores.  Connectors provide an easy and secure way to connect to external data without drivers or configuration. Direct connections use the same code you would use outside of Domino, with the flexibility to access files or data however you want.

**Domino Dataset:**
A Domino Dataset is a versioned, centralized data repository that enables teams to share, track, and manage data assets across projects and experiments. This ensures data consistency and reproducibility while providing governance controls and lineage tracking, eliminating the need for data scientists to manage their own copies of data.

**Compute Environments:**
Compute Environments are pre-configured, containerized environments that package all the tools, libraries, and dependencies needed for data science work. They enable instant reproducibility and portability of work across teams while eliminating environment setup overhead and "it works on my machine" issues.

**Hardware Tiers:**
Hardware Tiers are predefined compute resource configurations (CPU, GPU, memory) that users can select based on their workload requirements. This allows organizations to optimize costs by right-sizing resources for each task while giving data scientists flexibility to scale up for intensive computations without IT intervention.

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

## New Domino Concepts
**Jobs:**
Jobs are scheduled or on-demand executions of scripts, notebooks, or applications that run asynchronously in Domino without requiring an interactive session. They enable automated workflows, batch processing, and production deployment of models while providing full reproducibility, monitoring, and resource management capabilities.

**Domino Dataset Snapshots:**
Dataset Snapshots are immutable, point-in-time versions of Domino Datasets that capture the exact state of data at a specific moment. This enables perfect reproducibility of experiments and models by guaranteeing that the same data version can be accessed months or years later, while also supporting compliance and audit requirements.

**Artifacts:**
Artifacts are files or outputs generated during project executions (such as models, reports, or visualizations) that Domino automatically tracks and versions alongside the code and environment that produced them. This provides complete lineage and traceability of all project outputs, making it easy to reproduce results, share deliverables with stakeholders, and maintain governance over model assets.

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

## New Domino Concepts
**Domino Flows:**
Domino Flows is a visual workflow orchestration tool that allows users to build, schedule, and monitor complex data pipelines by connecting multiple tasks, jobs, and models in a directed acyclic graph (DAG). This enables teams to productionize end-to-end ML workflows with built-in error handling, dependency management, and automatic retries without writing orchestration code.

**Experiment Manager:**
Experiment Manager is a centralized tracking system that automatically captures and compares all experiment runs, including parameters, metrics, code versions, and results in a searchable interface. This accelerates model development by enabling data scientists to quickly identify the best-performing models, understand what changes improved performance, and reproduce any past experiment.

**Model Registry:**
Model Registry is a centralized repository that catalogs all trained models with their metadata, performance metrics, lineage, and deployment status throughout their lifecycle. This provides governance and collaboration capabilities by enabling teams to discover, compare, promote, and deploy models while maintaining full auditability and compliance documentation.

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


# 5.  Delivery and Hosting
In this phase, we will deploy the model for consumption by other users.  We will deploy the model as a REST API endpoint, and host a Streamlit app that calls the REST endpoint.

## Exercise Instructions


## New Domino Concepts
**Model Endpoints:**
Model Endpoints are REST API services that automatically deploy trained models as scalable, production-ready APIs with built-in load balancing, monitoring, and versioning capabilities. This enables data scientists to instantly serve predictions to applications and systems without writing deployment code or managing infrastructure, while IT maintains governance and security controls.

**Hosted Applications:**
Hosted Applications allow users to deploy and share interactive web applications (built with frameworks like Streamlit, Dash, or Flask) directly from their Domino projects with automatic scaling and authentication. This empowers data scientists to create self-service analytics tools and model interfaces for business users without requiring web development expertise or separate hosting infrastructure.

**Domino Launchers:**
Domino Launchers are customizable, self-service interfaces that allow business users to execute pre-configured data science workflows or models by simply filling out a form, without needing access to code or the full Domino platform. This democratizes access to data science outputs by enabling stakeholders to run analyses, generate reports, or get predictions on-demand while data scientists maintain control over the underlying logic and parameters.
 
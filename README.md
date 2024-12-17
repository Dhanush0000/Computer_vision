# Computer Vision Project

This project demonstrates a basic setup for a computer vision application using Flask. The following instructions guide you through the process of cloning, setting up, and running the project locally.

## Table of Contents
- [Cloning the Project](#cloning-the-project)
- [Setup Instructions](#setup-instructions)
  - [Install Flask](#install-flask)
  - [Create Project Structure](#create-project-structure)
  - [Create Basic Flask App](#create-basic-flask-app)
  - [Set Up Templates Folder](#set-up-templates-folder)
  - [Create HTML File](#create-html-file)
  - [Add Static Folder](#add-static-folder)
  - [Save Your Model](#save-your-model)
  - [Link Model in app.py](#link-model-in-app-py)
- [Running the Application](#running-the-application)

## Cloning the Project

To clone this project, use the following command:

```bash
git clone https://github.com/Dhanush0000/Computer_vision.git
```

## Setup Instructions

### Install Flask

Install Flask Using pip

```bash
pip install flask
```

### Create Project Structure

Create a main folder for your project:

```bash
mkdir flask_website
cd flask_website
```

### Create Basic Flask app

Inside flask_website, create the main Flask application file:

```bash
touch app.py
```

In app.py, initialize a basic Flask app.

### Set Up Templates Folder

Create a templates folder to store HTML files:

```bash
mkdir templates
```

### Create HTML File

Add an HTML file for the homepage in the templates folder (e.g., index.html).

### Add Static Folder

Create a static folder to contain CSS and JavaScript files:

```bash
mkdir static
```

Inside static, you can create css and js subfolders if needed.

It's only a if situation, index.html file has both css and js inline

### Save Your Model

Create a modules folder where you’ll save your trained model:

```bash
mkdir modules
```

Save your model file in this folder.

### Link Model in app.py

In app.py, provide the path to your saved model file for predictions. This will allow you to load the model and use it to predict classes.

## Running the Application

Once everything is set up, run the Flask application using the command:

```bash
python app.py
```

Check the terminal for a local link to access the app in your browser. Open the link to view and interact with your computer vision project.

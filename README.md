# ECS 171 Final Project - Automated Math Expression Solver (Group 18)

## Brief Introduction

![image](https://github.com/tsepten/mathExpressionSolver/assets/6781015/a628ce3f-880b-4880-abcb-91242be3de48)

We developed a simple automated handwritten math expression solver. The program can handle handwritten inputs that contain 0-9, +, -, /, \*, sqrt, (, ), and computes the answer of an expression through a four step process: (1) Image segmentation, (2) Image classification, (3) Equation parsing, (4) Equation solving.

## File/Folder Explanations
* _**Final project code executation demo.mp4**_ - This file (also uploaded on youtube, https://youtu.be/Dm57jEr4Frk ) shows the optional source code demo (code execution and training of CNN) of the whole program from start to end.
* _**171_finalproject.ipynb**_ - This is our main file and contains much of the work. However, this file imports models generated in the cnn.ipynb file.
* _**cnn_v2/cnn.ipynb**_ - This file contains our code for the convolutional neural network model and does exploratory data analysis on the dataset we train our CNN on.
* _**implementation_trials**_ - This file contains a lot of the code that we wrote during the iterative development process. It also contains old implementations that did not exactly work well.
* _**frontend**_ and _**backend**_ - This folder contains the frontend and backend for our demo process during presentation. Examples can be seen in the last few slides of Demo Slides.pdf
* _**hasyEXTENDED**_ - This folder contains infromation on our extension of the dataset. Note that the main hasyV2 dataset must be downloaded by the user, see 171_finalproject.ipynb
* _**testSegmentationImages**_ - This folder contains some sample images that our solver works well with.
* _**Demo Slides.pdf**_ - This file contains our pdf that we presented during presentation day
* _**Final Project Report.pdf**_ - This file is our 8-page final project report.
* _**Math Solver 1-pager.pdf**_ - This file is our initial plan we developed in the first few weeks. 
* _**Peer Reviews.pdf**_ - This file contains our peer reviews for other projects.
* _**Project Instructions.pdf**_ - This file contains some details on the guidelines/instructions for the final project.
* _**Project Roadmap.pdf**_ - This file contains the project roadmap and a very high level overview of how we should go about doing this project.

## How to run the code
There are two options to run the code: (1) Run the .ipynb files OR (2) run the demo (backend + frontend).
- (1) To run the .ipynb files, you need to go into the 171_finalProject.ipynb and simply run the cells. Ensure that the filepaths in the beginning are setup properly.
- (2) To run the demo, you need to go into type `python sever.py` on the backend folder. Make sure you have the necessary libraries. Then just open up index.html. 

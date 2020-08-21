# Avantari Technologies - Machine Learning Task Solution

Creating an AutoEncoder model for finding similar images and partitioning the dataset into K groups.

For detailed explanation please refer **README.pdf**.

## Sample outputs

The output from two different executions of the final solution notebook (**final_solution.ipynb**) is saved in HTML format for quick  viewing.

Please see the following files present inside **"Sample outputs"** directory:

* **final_solution_1_default.html**
* **final_solution_2_cached.html**

This can used to quickly see how final solution notebook execution looks like. Use this in case **final_solution.ipynb** fails to execute on your system for some reason.

## Requirements to execute *final_solution.ipynb* on your own

* Python 3.7.x or 3.8.x
* Jupyter Notebook ```pip install jupyter```
* Tensorflow 2.3.0 ```pip install tensorflow-gpu==2.3.0```
* Matplotlib ```pip install matplotlib```
* Pillow ```pip install pillow```

## Steps to execute

* Clone this GitHub repo: ```$git clone https://github.com/Shashank9830/avantari_tech```
* Change directory to avantari_tech: ```$cd avantari_tech```
* Load the solution notebook using ```$jupyter notebook final_solution.ipynb```
* Edit the variables in 3rd code cell before execution.
* Set the appropriate value for *input_file*, *mode* and *sim_count* (details are given in comments).
* Run all cells.
* For multiple executions, run all code cells below the 3rd code cell (including it) after making required changes to the 3rd code cell each time.

## Understanding final cell output
After execution of all cells, the final cell output should be like this:

![10 images similar to 46.jpg](https://github.com/Shashank9830/avantari_tech/blob/master/Sample%20outputs/img1.PNG "DEFAULT mode output")

![10 images similar to image 397](https://github.com/Shashank9830/avantari_tech/blob/master/Sample%20outputs/img1.PNG "CACHED mode output")

* Image in the first row in the **input image**.
* Images in the subsequent rows are **similar images** ranked in order of decreasing similarity.
* Similarity decreases from left-to-right and then top-to-bottom.

## Solution details

Refer **README.pdf** for detailed explanation of the approach used to find N similar images and to partition the dataset into K-groups.

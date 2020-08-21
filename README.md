# Avantari Technologies - Machine Learning Task Solution

Creating an **AutoEncoder** model for finding similar images and partitioning the dataset into K groups.

For detailed explanation please refer **README.pdf**.

## Sample outputs

![10 images similar to 46.jpg](https://github.com/Shashank9830/avantari_tech/blob/master/Sample%20outputs/img1.PNG "DEFAULT mode output")

The outputs from two different executions of the final solution notebook (**final_solution.ipynb**) is saved in HTML format for quick  viewing.

Please see the following files present inside **"Sample outputs"** directory:

* **final_solution_1_default.html**
* **final_solution_2_cached.html**

This can be used to quickly see how final solution notebook execution looks like. Use this in case **final_solution.ipynb** fails to execute on your system for some reason.

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

![10 images similar to image 397](https://github.com/Shashank9830/avantari_tech/blob/master/Sample%20outputs/img2.PNG "CACHED mode output")

* Image in the first row in the **input image**.
* Images in the subsequent rows are **similar images** ranked in order of decreasing similarity.
* Similarity decreases from left-to-right and then top-to-bottom.

## Solution details

Refer **README.pdf** for detailed explanation of the approach used to find N similar images and to partition the dataset into K-groups.

## File information.

|    | Filename                        | Type             | Information                                                                                        |
|----|---------------------------------|------------------|----------------------------------------------------------------------------------------------------|
| 1  | dataset                         | Directory        | Original dataset\.                                                                                 |
| 2  | resize\_dataset\.py             | Python    | Resizes the dataset images to 256x256\.                                                            |
| 3  | resized\_256                    | Directory        | Resized dataset\.                                                                                  |
| 4  | create\_autoencoder\.py         | Python    | Creates an autoencoder model\.                                                                     |
| 5  | autoencoder\.h5                 | H5          | AutoEncoder model saved in H5 format\.                                                             |
| 6  | trainer\_notebook\.ipynb        | Jupyter | Model training code\.                                                                              |
| 7  | trained\_autoencoder\.h5        | H5          | Trained autoencoder saved in H5 format\.                                                           |
| 8  | trained\_encoder\.h5            | H5          | Encoder part of the trained autoencoder\.                                                          |
| 9  | get\_encodings\.ipynb           | Jupyter | Code to get the encodings of all the images\.                                                      |
| 10 | encodings\.npy                  | NumPy       | Encodings of all 4738 images\.                                                                     |
| 11 | get\_similarity\.ipynb          | Jupyter | Code to find similarity of all the images with each other\.                                        |
| 12 | cosine\_similarity\_matrix\.npy | NumPy       | Cosine similarity matrix generated in the previous step\.                                          |
| 13 | sim\_mat\_sorted\.json          | JSON        | Images sorted in decreasing order of similarity to each other\.                                    |
| **14** | **final\_solution\.ipynb**          | **Jupyter** | **Main user notebook\. Run this for final output\.**                                                   |
| 15 | Sample outputs                  | Directory        | Some pre\-executed notebook outputs in HTML format\. One example of both cached and default mode\. |
| 16 | k\_grouping\.py                 | Python    | Code to implement Elbow and K\-medoids algorithm\.                                                 |
| 17 | k\_groups\.json                 | JSON        | JSON file containing list of medoids and clusters                                                  |
| 18 | partition\_dataset\.py          | Python    | Code to partition the dataset as mentioned in the above JSON file                                  |
| 19 | K Groups                        | Directory        | Folder containing K\-Groups                                                                        |
| 20 | \.ipynb\_checkpoints            | \-\-\-           | \-\-\-                                                                                             |

## Authors

* **Shashank Singh** - *Complete work* - [shashank9830](https://github.com/shashank9830)

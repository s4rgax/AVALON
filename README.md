# AVALON
### An attention-based CNN approach to detect forest tree dieback caused by insect outbreak in Sentinel-2 images

## Abstract
Forests play a key role in maintaining the balance of ecosystems, regulating climate, conserving biodiversity, and supporting various ecological processes. However, insect outbreaks, particularly bark beetle outbreaks, pose a significant threat to European spruce forest health by causing an increase in forest tree mortality.  Therefore, developing accurate forest disturbance inventory strategies is crucial to quantifying and promptly mitigating outbreak diseases and boosting effective environmental management.
In this paper, we propose a deep learning-based approach, named AVALON, that implements a  CNN to detect tree dieback events in Sentinel-2 images of forest areas.
To this aim, each pixel of a Sentinel-2 image is transformed into an imagery representation that sees the pixel within its surrounding pixel neighbourhood. We incorporate an attention mechanism into the CNN architecture to gain accuracy and 
achieve useful insights from the explanations of the spatial arrangement of model decisions.
We assess the effectiveness of the proposed approach 
in two case studies regarding forest scenes in the Northeast of France and the Czech Republic, which were monitored using Sentinel-2 satellite in October 2018 and September 2020, respectively. Both case studies host bark beetle outbreaks in the considered periods.

## Installation

1. Clone the repository: `git clone 'https://github.com/xxxxxx/AVALON.git/`
2. Install dependencies: `pip install -r requirements.txt`

## Usage
1. Download data:
    first of all you havo to download all needed data from https://mega.nz/folder/zSIEyLoZ#22eOMKinl_mCIo2NFQOCLw and place Sentinel 2 and Mask data in your project folder.
2. Set all the configuration variables in conf/Sentinel2.conf file, setting the path where you placed the mask and Sentinel data.


Now you can run the experiments with the following command:
```bash
python src/main.py dataset1 conf/Sentinel2.conf
```

#### Parameter description
1. The first parameter is for selecting the dataset. You can declare much dataset in conf file, so you can customize all the paths and names.
The section to describe the dataset configuration is declared using reflection mechanism, so in this example the Czech Republic is named dataset1, and we use dataset1 param to start the train on this dataset (same for france named dataset2).
2. The second parameter is to refer the configuration file. Ypu can also create different customized configuration files and get it into the second parameter. 
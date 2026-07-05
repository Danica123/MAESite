MAESite
====

Environment Setup
----
The virtual environment is set up as follows:  
* ESM3 (https://github.com/evolutionaryscale/esm)  
* Python==3.10.4  
* numpy==1.22.4   
* pandas==2.2.3  
* pytorch==1.12.0  

Set config
----
The "optimized_train_config.py" file should be set up correctly according to your software environment:  
* parser.add_argument('--indir', type=str, default="/pubssd/dhh/MAESite/DNA_train_data/")    
* parser.add_argument('--save_dir', type=str, default="/home2/2023/23dhh2/MAESite/model/DNA")   

Experimental Procedure
----

* Step 1: Activate the virtual environment and run `Python train.py` for training.   
* Step 2: run `Python predict.py` for testing.   

Note
-----
Have a nice day!  

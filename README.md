Behrooz's neural machine translation toolkit
============================================

Main goal
---------
In exploring the plethora of tools for mapping sequences, I could not find a 
simple and yet accurate implementation of Transformer model. This repository
contains results of my effort for reimplementing this model simply and yet
accurately.
I will base all further research projects during the course of my Ph.D. on the
basis of this code.

Example commands
----------------
For training a new model
```
python3 -m nmt -m train -c <path_to_base_config_file> [-t <model_name_to_override_config>]
```
For translating a text file
```
python3 -m nmt -m translate -c -c <path_to_base_config_file> -i <input_text_file_path> -o <output_file_path> [-t <model_name_to_override_config>] [-x <experiment_index>]
```
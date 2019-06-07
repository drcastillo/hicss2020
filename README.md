# hicss2020
Repo for the work required for the HICSS conference

# Getting Started
Main work is done in the notebooks labelled 'explainability'

These notebooks have many dependencies which making cloning a tad bit tedious.
A few ways to go about this:
- 1:
  - Clone the repo directly from github
    - Open up the lending club explainability notebook.
      - Uncomment-out the first line & run the first block of code. This pip installs all local dependencies from my environment:
        ```python
        !pip install -r requirements.txt --user
        ```

- 2:
  - Clone the repo directly from github
    - Go into conda prompt and change directory to the repo path:
    '''
    cd <path>
    '''
      - run the following to create a new conda virtual environment, building on requirements.text
      ```python
      #Your environment name goes inside brackets
      conda create --name <env_name> --file requirements.txt
      activate <env_name>
      ```
        - Launch Jupyter notebook within the activated environment
  - 3:
   - The First 2 blocks of code deal with functions with global dependencies and the loading of all data objects.
    Objects were stored locally during model training to cut down on load times.

# Brief Function Descriptions
Most functions were designed using Plotly to add interactivity and lessen dependency on manual parameter entry.
A few examples. If you want to look at the classification reports for each model, as judged by inference on test set, toggle through the models:
Inline-style:
![alt text](https://github.com/drcastillo/hicss2020/blob/master/images/classreport_jpg.PNG "Logo Title Text 1")

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

# GBM resectability prediction

This repository contains the source code for the publication "Improved Prediction of Surgical Resectability in Patients with Glioblastoma using an Artificial Neural Network" by Adam P Marcus, Hani J Marcus, Sophie J Camp, Dipankar Nandi, Neil Kitchen, and Lewis Thorne.

## Usage

To ensure reproducibility the entire project runs within a Docker container. If needed, please [install Docker](https://docs.docker.com/get-docker/) before proceeding. Building, testing, and then running the project can be done as follows:

```bash
docker build -t gbm-resectability-prediction .
docker run -it -v $(pwd):/home/ gbm-resectability-prediction build test run
```

Please note that [FANN formatted](https://libfann.github.io/fann/docs/files/fann_training_data_cpp-h.html#training_data.read_train_from_file) training data files are required and should be placed under `./data/raw/`. Ethics and privacy concerns prevent sharing of the original data set.

## Contributing

Contributions are welcomed! The project's structure is based on [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/). All C++ code should adhere to the [Google Style Guide](https://google.github.io/styleguide/cppguide.html) with two allowed exceptions: frequent use of unsigned integers (to facilitate integration with the FANN library), and lack of namespaces (to shorten identifiers as small project and clashes are unlikely). Comments should be [compliant with Doxygen](http://www.doxygen.nl/manual/docblocks.html).
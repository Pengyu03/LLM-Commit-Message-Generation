# Automated Commit Message Generation with Large Language Models: An Empirical Study and Beyond
This repository contains our code and dataset for studying the performance of Large Language Models (LLMs) in generating commit messages. Our research covers datasets from five programming languages (Java, Python, C#, C++, JS), along with code for generating commit messages using different LLMs (LLaMA-7B, LLaMA-13B, GPT-3.5, Gemini).

## Datasets:
We provide cleaned datasets for the following five programming languages:
* Java
* Python
* C#
* C++
* JavaScript

The datasets are located in the `datasets` directory, with each language's dataset stored in its respective subdirectory.

## Code:
This repository includes the following code:

* Code for reproducing the Bi-LSTM model: Located in the `models/Bi-LSTM` directory.
* Code for LLM-based commit message generation: Located in the `models/LLM` directory, comprising code for LLaMA-7B, LLaMA-13B, GPT-3.5, and Gemini models. For GPT-3.5 and Gemini, we provide two versions of the code: one with examples (with_example) and one without examples (without_example).
* Tools for retrieving the best examples: Located in the `tools` directory, including semantic retrieval tools and lexical retrieval tools.

## Quick Start

### Datasets
The datasets have been preprocessed and can be directly used for model training and testing.

### Reproducing the Bi-LSTM Model
Navigate to the Bi-LSTM directory and execute the following code to reproduce the Bi-LSTM model:

#### Step 1: `filter`: Code for data preprocessing according to the input requirements of the Bi-LSTM model.

- Configuration Variables

  Before running the code, please modify the following variables according to your requirements:

  * `input_file`: Specify the file path of the input data, for example, `input.jsonl`.
  * `lan_output_file`: Specify the output file path for language data, for example, `output.jsonl`.
  * `other_output_file`: Specify the output file path for other data, for example, `other_data.jsonl`.
  * `lan`: Specify the programming language being processed, for example, `Java`.

- Running the Code

  Ensure that you have correctly set the variables mentioned above.

  Run the corresponding Python script to perform the task, for example:
  ```python
  $ python filter.py
  ```
#### Step 2: Training the Model

- Configuration Variables

  Before running the code, please modify the following variables according to your requirements:
  
  * `batch_size`: Specify the batch size, for example, `128`.
  * `epochs`: Specify the number of training epochs, for example, `10`.
  * `dropout`: Specify the dropout rate, for example, `0.4`.
  * `rnn_hidden`: Specify the size of the RNN hidden layer, for example, `768`.
  * `rnn_layer`: Specify the number of RNN layers, for example, `1`.
  * `class_num`: Specify the number of classes, for example, `4`.
  * `lr`: Specify the learning rate, for example, `0.001`.
  * `train`: Specify the file path of the training data, for example, `./data/archive/train_clean2.csv`.
  * `val`: Specify the file path of the validation data, for example, `./data/archive/val_clean.csv`.
  * `test`: Specify the file path of the testing data, for example, `./data/archive/test_clean.csv`.

- Running the Code

  Ensure that you have correctly set the variables mentioned above. Run the corresponding Python script to perform the training task, for example:
  ```python
  $ python whatwhytrain.py
  ```

#### Step 3: Model Testing:

- Configuration Variables

  Before running the code, please modify the following variables according to your requirements:

  * `PATH`: Specify the path to the model checkpoint, for example, `./lightning_logs/version_1/checkpoints/epoch=29-step=330.ckpt`.
  * `batch_size`: Specify the batch size, for example, `128`.
  * `epochs`: Specify the number of testing epochs, for example, `30`.
  * `dropout`: Specify the dropout rate, for example, `0.4`.
  * `rnn_hidden`: Specify the size of the RNN hidden layer, for example, `768`.
  * `rnn_layer`: Specify the number of RNN layers, for example, `1`.
  * `class_num`: Specify the number of classes, for example, `4`.
  * `lr`: Specify the learning rate, for example, `0.001`.
  * `num_number`: Specify the number of batch files for testing data, for example, `5`.
  * `test`: Specify the file path of the testing data, for example, `./data/archive/cstest/test{num}.csv`.
  * `f`: Specify the output file path for prediction results, for example, `preds_csharp.json`.

- Running the Code

  Ensure that you have correctly set the variables mentioned above. Run the corresponding Python script to perform the testing task, for example:
  ```python
  $ python whatwhytest.py
  ```
### LLM Commit Message Generation:

#### Configuration Variables
Before running the code, please modify the following variables according to the version and type of LLM:
- With Example (with_example):

  * `lan`: Specify the path of the language data file, for example, `java.jsonl`.
  * `output_filenames`: A dictionary specifying the paths of multiple output files, where each output file corresponds to different numbers of examples (1, 3, 5, 10). For example, `pyaddresult_gemini_1nov.jsonl`.
  * `best_file`: Specify the path containing the file with the best examples.
  * `key`: In Gemini's running code, fill in your Gemini API key.
  * `key_list`: For GPT-3.5's running code, a list containing multiple keys can be used. You can fill in a certain number of GPT API keys according to your actual needs.

- Without Example (without_example):

  * `lan`: Specify the path of the language data file, for example, `java.jsonl`.
  * `output_filename`: Specify the output file path, for example, `java_result_7B.jsonl`.
  * `key`: In Gemini's running code, fill in your Gemini API key.
  * `key_list`: For GPT-3.5's running code, you can fill in a certain number of GPT API keys according to your actual needs.

#### Running the Code
Ensure that you have correctly set the variables mentioned above. Run the corresponding Python script to perform the commit message generation task, for example:
```python
$ python GPT_with_example.py
```

### Calculate Automated Evaluation Metrics:

- Configuration Variables

  Before running the code, please modify the following variables according to your requirements:

  * `file`: Specify the file path containing the generated text, for example, `csgptnoexample.jsonl`.
  * `nlp_file_path`: Specify the file path containing the reference text for automated evaluation metrics, for example, `gptjavainnlp.jsonl`.

- Running the Code

  Ensure that you have correctly set the above variables. Run the corresponding Python script to perform the task of calculating automated evaluation metrics, for example:
  ```python
  $ python metrics.py
  ```

### Retrieve the Best Examples:
Enter the `tools` directory and choose the appropriate retrieval tool:

#### Lexical Retrieval of Best Examples
- Configuration Variables

  Before running the code, please modify the following variables according to your requirements:

  * `lan`: Specify the path of the language data file, for example, `java.jsonl`.
  * `train`: Specify the path of the training data file, for example, `javatrain.jsonl`.
  * `output_filename`: Specify the path of the output file, for example, `java_with_best.jsonl`.

- Running the Code

  Ensure that you have correctly set the above variables. Run the corresponding Python script to perform the lexical retrieval task, for example:
  ```python
  $ python lexical_retrieval.py
  ```

#### Semantic Retrieval of Best Examples
Semantic retrieval consists of two steps, vectorization, and finding the best examples:

- Step 1: Vectorization:

  - Configuration Variables
  
    Before running the code, please modify the following variables according to your requirements:

    - `lan`: Specify the path of the language data file, for example, `py1.jsonl`.
    - `output_file`: Specify the path of the output vector data file, for example, `vpy1no.jsonl`.

  - Running the Code
  
  Ensure that you have correctly set the above variables. Run the corresponding Python script to perform the vectorization task, for example:
  ```python
  $ python semantic_retrieval_vectorization.py
  ```

- Step 2: Finding the Best Examples:

  - Configuration Variables
  
    Before running the code, please modify the following variables according to your requirements:
  
    - `vlan`: Specify the path of the language vector data file, for example, `vpy1no.jsonl`.
    - `vtrain`: Specify the path of the training vector data file, for example, `encoded_diffspy2.jsonl`.
    - `output_file`: Specify the path of the output file containing the retrieved best matches, for example, `pybest_no_selectv.jsonl`.
    - `input_file`: Specify the file path being retrieved, for example, `pytrain_no_selectv.jsonl`.

  - Running the Code
  
  Ensure that you have correctly set the above variables. Run the corresponding Python script to perform the semantic retrieval task, for example:
  ```python
  $ python semantic_retrieval_find.py
  ```

## Contribution
Contributions to this project through Pull Requests or Issues are welcome.

## License
This project is licensed under the `MIT` License.

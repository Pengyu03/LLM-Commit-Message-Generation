# Automated Commit Message Generation with Large Language Models: An Empirical Study and Beyond
This repository contains our code and dataset for studying the performance of Large Language Models (LLMs) in generating commit messages. Our research covers datasets from 8 programming languages (Java, Python, C#, C++, JS, Rust, Go, PHP), along with code for generating commit messages using different LLMs (LLaMA-7B, LLaMA-13B, GPT-3.5, Gemini). 

Our paper has been published in [IEEE Transactions on Software Engineering](https://ieeexplore.ieee.org/document/10713474)\, more details are provided in our paper. [Click here to view PDF](https://arxiv.org/pdf/2404.14824)

## Datasets:
We provide cleaned datasets for the following 8 programming languages:
* Java
* Python
* C#
* C++
* JavaScript
* Rust
* Go
* PHP

The datasets are located in the `datasets` directory, with each language's dataset stored in its respective subdirectory.

The following is the scale table of MCMD+ constructed in this study:
![460369cb1c33782ead546c9d0de35e8](https://github.com/user-attachments/assets/b8f0f4bb-34c4-480f-8d04-c93928031db2)



## Code:
This repository includes the following code:

* Code for reproducing the Bi-LSTM model: Located in the `Bi-LSTM` directory.
* Code for LLM-based commit message generation: Located in the `Models & Metrics` directory, comprising code for LLaMA-7B, LLaMA-13B, GPT-3.5, and Gemini models. For GPT-3.5 and Gemini, we provide two versions of the code: one with examples (with_example) and one without examples (without_example).
* Tools for retrieving the best examples: Located in the `tools` directory, including semantic retrieval tools and lexical retrieval tools.

## Quick Start

### Datasets
The datasets have been preprocessed and can be directly used for model training and testing.

### ERICommitter Web
**ERICommiterWeb** is a web-based commit message generator that helps developers automatically generate high-quality commit messages to improve the efficiency of code management. The following is ERICommiterWeb's user interface:
![User interface](https://github.com/user-attachments/assets/6b6ec46f-017b-4d84-a916-20b2537fa0e4)

### Prerequisites

Before starting, ensure you have the following ready:

- **Server Environment**: A Linux server (e.g., Ubuntu).
- **Git**: For version control and code management.
- **Python 3.x**: The runtime environment required for the application.
- **Nginx**: A web server.
- **Systemd**: For service management.
- **CodeReviewer Model**: Download the pre-trained [codereviewer](https://huggingface.co/microsoft/codereviewer) model file.
- **GPT-3.5 API Key**: You need to obtain an API key from [OpenAI](https://platform.openai.com/) to enable GPT-3.5 functionalities.
- **Vector Retrieval Libraries**:
  - For Python, ensure the file `vtrain_py.jsonl` is prepared.
  - For Java, ensure the file `vtrain_java.jsonl` is prepared.
- **Commit Message Original Text Files**:
  - For Python commits, prepare the file `pythontrainuyuan3.jsonl`.
  - For Java commits, prepare the file `javatrainyuan3.jsonl`.


### Server Configuration from Scratch

#### 1. Install Required Software

First, ensure that Git, Python 3, and Nginx are installed on your server.

```bash
sudo apt update
sudo apt install git python3 python3-venv python3-pip nginx
```

#### 2. Clone Project Code
Use Git to clone the ERICommiterWeb project code onto the server.

```bash
cd /var/www/
sudo git clone https://github.com/Pengyu03/LLM-Commit-Message-Generation.git
cd ERICommiter_Web
```

#### 3. Set Up Virtual Environment and Install Dependencies
Create and activate a Python virtual environment, then install the required dependencies.

```bash
python3 -m venv venv
source venv/bin/activate
```
#### 4. Configure Systemd Service
Create a new Systemd service file to manage the application using systemctl.

```bash
sudo nano /etc/systemd/system/ericommiterweb.service
```

Add the following content to the file:


```ini
[Unit]
Description=ERICommiterWeb Service
After=network.target

[Service]
User=www-data
WorkingDirectory=/var/www/ERICommiterWeb
ExecStart=/var/www/ERICommiterWeb/venv/bin/python3 /var/www/ERICommiterWeb/app.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Save and exit the editor, then enable the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable ericommiterweb
```

#### 5. Configure Nginx
Edit the Nginx configuration file to proxy traffic to the application.

```bash
sudo nano /etc/nginx/sites-available/ericommiterweb
```
Add the following configuration:

```nginx
server {
    listen 80;
    server_name your_domain_or_IP;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

**Note:** In order to ensure that the program is used by SDUOJ team members during the study, the Site Key is set as `sduoj`. You can continue to use the Site Key or change the program to set your Site Key. 

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

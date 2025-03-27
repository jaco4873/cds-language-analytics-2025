# Assignment 3

## Generative n-gram language models

This assignment is about creating the machinery:
1. For training a generative n-gram language model 
2. For using that same language model to generate text

You are expected to have at least two scripts: `train.py` and `generate.py`. It will likely also make sense to have at least one common module in which you have functionality needed by both of the main scripts.

These scripts should work as scripts that you run from the command-line with appropriate arguments. That is, one should be able to do something like this:

```bash
# first, I train a model on my Gutenberg data with 3-grams
python train.py gutenberg-model data/gutenberg --n-gram-size 3

# then, I generate a 100 word text with the model using top-k sampling of 25
python generate.py gutenberg-model --tokens 100 --top-k 25
```

The `run.sh` script should do something like the thing above to demonstrate that it works. Feel free to alter it how you see fit.

How exactly your command-line interface should work is up to you, e.g. whether you want named arguments or positional arguments.

## Suggested arguments for scripts

Below are suggested arguments for your scripts. You can also add more if you want!

### train.py

| Suggested name | Purpose | Mandatory for the assignment? |
|---------------|---------|------------------------------|
| model-name | An identifier for the model such that you can save and load it using that identifier | Yes |
| data-path | Where the script should find the training data | Yes |
| n-gram-size | A number to determine the size of the n-grams for your n-gram model | Yes |
| smoothing | A signal for your script about "smoothing" counts or not | No |
| stupid-back-off | A signal for your script whether it should create lower-order n-gram models for doing stupid backoff | No |

### generate.py

| Suggested name | Purpose | Mandatory for the assignment? |
|---------------|---------|------------------------------|
| model-name | An identifier for the model that the script should use for generating text | Yes |
| tokens | The number of tokens that your script should generate | Yes |
| seed | An (optional) starting seed for text generation | No |
| top-k | An optional argument to enforce top-k sampling | Yes |
| top-p | An optional argument to enforce top-p sampling | No |
| temperature | An optional argument to lower or raise the temperature for the conditional probability distributions | No |

## Objectives

This assignment is designed to test that you can:

- Create multiple scripts that draw on common functionality
- Integrate the use of command-line arguments for your scripts
- Use common data structures in Python to solve complex problems
- Generalize your code to not depend on specific data or use-cases

## Notes and hints

- For command-line arguments, use the package `click`, and make the user experience great
- To pack together functionality that belongs together, defining your n-gram model as a class may help in structuring your code nicely. See further down for a suggested template to get you started.* It is no requirement to do that, though. You can also define it in a handful of functions instead, which is just as valid an approach.
- For the actual training and generation, most code that you will need can be found in the n-gram language model notebook.
- If you decide to integrate e.g. both top-k and top-p sampling, make sure that your code handles that they cannot work together. The same goes for other arguments that may conflict.

\* A suggested template to get you started on creating an NgramModel class:

```python
class NgramModel:
    def __init__(self, name: str, n_gram_size: int = 2):
        self.name = name
        # further initialization

    def train(self, folder_path: str):
        """Train model on *.txt files in the folder."""
        # do training stuff

    def conditional_probability_distribution(self, history: tuple[str]):
        """Return p(w|history)"""
        # get and return the appropriate distribution from the internal model
        # handle cases where it is not there

    def generate(self, seed: tuple[str], tokens: int = 25, top_k: int = None):
        """Generate a number of tokens given a seed.
        top_k is an optional argument to do top-k sampling."""
        # given the seed argument, start generating tokens

    def save(self, models_path: str = "models/"):
        """Save the model."""
        # find a suitable way to store your model

    @classmethod
    def load(cls, model_name: str, models_path: str = "models/"):
        """Load and return an existing model"""
        # this should result in an NgramModel object being returned.
        # a class method belongs to the class, not the object;
        # you can then do something like this in your code:
        # ngram_model = NgramModel.load("gutenberg-model")
        # but there are also other ways to approach this!
```

## Assignment Submission Requirements

Your submission should be a `.zip` file containing all necessary components to run your code. The following elements are required:

### Required Files and Structure

- `pyproject.toml`
  - Lists all project dependencies (and general project specification as a typical project file)

- `setup.sh`
  1. Creates a virtual environment (using UV and the command UV sync)
  2. Installs dependencies (via UV SYNC as well)
  3. Handles any additional setup 

- `src/` folder
  - Contains all `.py` files organized following good software principles
  - Should contain a settings.py file, which uses Pydantics BaseSettings. This will be the default way to set arguments for the run and settings, but can be overridden by the command line (that should otherwise default to any settings set here)
  - Contains a main.py file as the computational entry point to the application invoked in run.sh
  
- `run.sh`
  - Executable via `./run.sh`
  - By default, runs `src/main.py`

- `data/` folder
  - Contains all required data files
  - Include data even if available on UCloud

- `output/` folder
  - Destination for output files
  - Can be created dynamically by your code

- `README.md`
  - Serves as project documentation
  - Should include:
    - Project purpose
    - Project structure
    - Installation instructions
    - Running instructions
    - Code procedure/workflow
    - Any other relevant documentation
  - Acts as the entry point to your project
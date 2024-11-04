# Classes

## Agent
- action space
- enforce some sort of action validation (TBD)
- Prompts
  - system prompt
  - observation prompt (how do we format the observation that the agent gets from the environment?)

## Action ()
- defines an action as a python dataclass
- options to format to a string from the class
- can provide better validation of agent outputs if necessary

## EnvironmentWrapper (Sebastian)

- wraps existing environments with a unified interface, will allow us to run multiple environments.
- Benchmarking scripts will use this class.
- provides the Database class to save data from the environment
- defines the tables used in the database
- automates creation of multiple agents

## Database
- allows simple operations on a local sqlite database (Select, Delete, Update, Insert).
- static table configurations inside the class
- options to create a new .db file on every run, or to continually add data to the database.

## Fine-tuning
- OpenAI fine-tuning
- 

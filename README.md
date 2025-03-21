# Natural language processing course: `Automatic generation of Slovenian traffic news for RTV Slovenija `
Authors: [**`Miha Lazić`**](https://github.com/Lazzo23), [**`Luka Gulič`**](https://github.com/GulicLuka), [**`Matevž Crček`**](https://github.com/Matslo1234)

## About
Traffic reporting is a crucial component of public broadcasting, providing real-time updates on road conditions, accidents, and congestion. At RTV Slovenija, traffic news is currently compiled manually by students who review traffic reports from the [_promet.si_](https://www.promet.si/sl) portal and transcribe them into structured news segments, which are then broadcast every 30 minutes. This manual approach is time-consuming, prone to human error, and inefficient given the volume of traffic data that needs to be processed. 

With the advancement of large language models (LLMs), there is an opportunity to automate the generation of structured and readable traffic news. This project aims to leverage an existing LLM, apply prompt engineering techniques and fine-tune the model to automatically generate traffic reports that adhere to the broadcasting standards of RTV Slovenija. In the next chapter, we present various existing solutions that we consider using in this project.

## Project Structure
**`/Code`** contains project source code.  
**`/Data`** contains project raw data (used for fine-tuning, evaluation, ...).  
**`/Images`** is output folder for project source code.  
**`/Instructions`** contains given project instructions and traffic report rules.  
**`/Report`** contains temporary project report written in LaTeX.  

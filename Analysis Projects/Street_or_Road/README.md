# Activity: Street or Road?

## Instructions 

- Complete a modification of the **Street or Road?** Activity found in the Data Computing eBook
    - **IMPORTANT: The eBook instructs you to use either the voter addresses OR the medicare service provider addresses--you need to use BOTH for the "Back to the Streets" section**
    - Specifically, **BEFORE** starting the "Back to the Streets" section: 
        1. a comparison of the two address sets (i.e., how many addresses match in both data sources)
        2. you will then need to combine the voter addresses & the medicare service provider addresses (e.g., using `bind_rows( )`). Before you bind the two datasets, using mutate function add a column source to each dataset mentioning the datasource. (e.g., source for 1st dataset is Wake County and 2nd dataset is Medicare Services)
        3. lastly, after refining your Regular Expression process to capture popular street endings you are asked to "Present your results."  You should compare and contrast the types of street endings common among the two address sources (e.g., similarities and differences among common street endings for voters & medicare service providers)
    - make sure you carefully follow the instructions and mirror the code in the activity as you work 
    - **many tasks in the activity should have narrative text describing your observations; most steps also require code chunks and corresponding output.**
- Each student will submit the activity individually to Canvas, but students are encouraged to help one another complete the work
- submit the completed R Notebook to Canvas before deadline


## Grading

##### The assignment is worth a total of 15 points.

- Turn in your HTML R Notebook in Canvas before the due date.
- [1 point] Before starting the "Solved Example" State how many addresses are found in BOTH (i.e. How many common addresses are there?)
- [2 points] Work through the "Solved Example" section showing progress with each step
- "Back to the Streets" Section:
    - [2 points] Your Turn #1: explain each line of code in English (either narrative or commented code)
    - [2 points] Your Turn #2: expand to match several more patterns (at least 12 total)
    - [2 points] Your Turn #2: provide a table in **descending order** of popularity for the street name identifiers you found for the combined address pool of Wake County voters and medicare service providers
    - [4 points] Your Turn #2: use `ggplot` to construct a bar chart in **descending order** of popularity for the street name identifiers you found for each address source. (i.e. you need 2  bar  charts)
    - [2 points] Write a few sentences to compare and contrast the types of street endings common among the two address sources (e.g., similarities and differences among common street endings for voters & medicare service providers)


##### Tips: 

**IMPORTANT: The eBook instructs you to use either the voter addresses OR the medicare service provider addresses--you need to use BOTH**

- *Note: There's nothing to do in the "For the professional..." section at the very end except to be impressed.*


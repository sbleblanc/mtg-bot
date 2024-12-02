# mtg-bot
The big picture idea is to build a simple RAG system with a model like Phi 3.5 that can easily run locally and that is good enough to answer questions about Magic The Gathering rules and ideally explain weird interactions like Opalescence and Humilty.

It is true that Magic the Gathering is big game and a lot of information on the game is available online, hence modern LLMs have some knowledge about MTG. Without any additional context, ChatGPT is able to answer questions about tricky (even for humans) card interactions such as Opalescence and Humility which involves the layering system dictating the order in which effects are applied. Fortunately, this doesn't hold for smaller models like phi 3.5 mini for example which can be broken with simple questions like "In Magic The gathering, what is the maximum amount of cards that you can have in your deck?". I know LLMs are usually not good at giving factually correct answers in general, so this is why asking such a question to a LLM and seeing it react as expected is just a sanity check to ensure the project has merit. This also means it's a good candidate model to explore RAG applications since I know for sure I can't trust its "implicit knowledge".

## Data

The intended sources of data for the bot are the rules and all the cards information(which includes specific rulings that occurred throughout MTG's history for ambiguous cards). 

### Rules
The rules can be taken from the official MTG website in TXT format. By its nature, the rules are split in 1-line paragraph, which simplifies the preprocessing. Each rules is considered a document as it didn't make sense to me to chunk the text as if the whole rulebook was a big continuous block of text. Also, since alls rules have a section/subsection number, it makes it so the LLM can add a precise reference to a rule when it's present in its context.

### Cards
I didn't add the logic to process that cards yet. I was intending to use the MTG JSON project to get the information and ideally I was going to consider each card as its own document similar to the rules as a think they are small enough and independent entities that the text embeddings will still make sense.

# cfv-price-model
This project is a model trained to guess the online selling prices of Cardfight!! Vanguard trading cards, combining structured card attributes with card effect text via a dual-encoder (DistilBERT + numeric MLP) architecture.

## Data Source
Card and pricing data sourced from [TCGCSV](https://tcgcsv.com/).

## AI-Generated Code
Portions of this project's code were generated with the assistance of AI tools.

## How to Use
- Place any TCGCSV data you would like to train on into card_data/sets/ (this model is currently designed for use with D-Standard cards only).
- Then, run *csv_handler.py*, which will combine all the card sets into a file named *compiled_data.csv*.
- Then, run *train.py* which will train the model on your chosen data. The hyperparameters can be changed using the following environment variables:
    - EPOCHS_STAGE1: epochs of the numeric training phase
    - EPOCHS_STAGE2: epochs of the fine-tuning phase
    - LEARNING_RATE: lr for the numeric/fusion layers in both stages, DistilBERT uses 1% of this rate in the stage 2 fine-tuning phase
    - DROPOUT: dropout ratio
    - BATCH_SIZE
    - HIDDEN_SIZE
- Now you can run inference using *infer.py* to guess the price of any potential card with an existing nation and rarity.

### *infer.py* example:

python infer.py --nation "Dark States" --rarity R --grade 1 --shield 5000 --critical 1 --power 8000 --text "Placeholder Card Name: During your turn, if you have a grade 3 or greater vanguard, this unit gets [Power]+5000."

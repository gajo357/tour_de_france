Small project that collects data from procyclingstats.com, saves it, trains an ML model, and tries to predict the performance at the 2017 Tour de France race.
Made for fun, to compete with coleagues in a manager game at https://www.holdet.dk/da/tour-de-france-2017/

The code could have better naming and more comments, but it was done in a hurry.
The methods for collecting data are in pro_cycling_scraper.py, for training in training_model.py, and for prediction in prediction.py.
The final model is saved in model.pkl.

The model tries to predict the ranking of the player in each stage based on player's and stage's features.
In the end, we sort the players based on their value for money index (1/(ranking*cost)).
In this way, I tried to get the most points for the limited amount of money available in the manager game.

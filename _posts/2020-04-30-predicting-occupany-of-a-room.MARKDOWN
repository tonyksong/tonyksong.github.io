---
layout: post
title:  "Predicting Occupancy of a Room"
date:   2020-04-30 18:34:43 -0400
categories: jekyll update
---
I decided to take on the [Occupancy dataset from UCI Machine Learning][occupancy]. With this dataset, I wanted to accomplish 2 tasks:

1. Predict the occupancy of a room given parameters.
2. Conduct Time Series Analysis to forecast the parameters and predict occupancy.

The tasks may sound similar, but they're actually different. The first one does not take the time into account, but rather uses the parameters or a subset of the parameters given (Light, Humidity, Temperature, CO2, etc.) to predict the occupancy of the room (binary -> 0 or 1). The second task actually does take the time into account, and forecasts the individual parameters to predict the occupancy of the room at specific times where the parameters were forecast.

[occupancy]: https://archive.ics.uci.edu/ml/datasets/Occupancy+Detection+
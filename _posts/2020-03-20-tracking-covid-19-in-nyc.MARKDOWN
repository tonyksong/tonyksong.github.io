---
layout: post
title:  "Tracking COVID-19 in NYC"
date:   2020-03-20 18:34:43 -0400
categories: covid-19
---
With the world going through the COVID-19 pandemic, and given that I'm in NYC, which is the epicenter of the outbreak here in the U.S., I thought it would be a fun exercise to track the number of cases and deaths in NYC. There are a lot of great visualizations out there, and I've been checking the [COVID-19 Dashboard by JHU CSSE][jhu-viz], but I wanted an easier, faster way to just see the information for NYC.

I found a data source from New York Times that gets updated daily on GitHub (<https://github.com/nytimes/covid-19-data>). From this data, I used D3.js (which I learned in my course CSE 6242 Data Analytics and Visualization at Georgia Tech) to create a simple interactive plot that tracks the number of confirmed COVID-19 cases and deaths in NYC. Check it out [here!][covid19nyc]

<!-- {::nomarkdown}
<iframe src="https://tonyksong.github.io/covid19nyc.html" marginwidth="0" marginheight="0" scrolling="no" width="100%" height="100%"></iframe>
{:/} -->

[jhu-viz]: https://www.arcgis.com/apps/opsdashboard/index.html#/bda7594740fd40299423467b48e9ecf6
[covid19nyc]: https://tonyksong.github.io/covid19nyc.html

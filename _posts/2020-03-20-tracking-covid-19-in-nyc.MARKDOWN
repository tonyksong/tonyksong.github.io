---
layout: post
title:  "Tracking COVID-19 in NYC"
date:   2020-03-20 18:34:43 -0400
categories: covid-19
---
With the world going through the COVID-19 pandemic, and given that I'm in NYC, which is the epicenter of the outbreak here in the U.S., I thought it would be a fun exercise to track the number of cases and deaths in NYC. There are a lot of great visualizations out there, and I've been checking the [COVID-19 Dashboard by JHU CSSE][jhu-viz], but I wanted an easier, faster way to just see the information for NYC.

I found a data source from New York Times that gets updated daily on GitHub (<https://github.com/nytimes/covid-19-data>). From this data, I used D3.js (which I learned in my course CSE 6242 Data Analytics and Visualization at Georgia Tech) to create a simple interactive plot that tracks the number of confirmed COVID-19 cases and deaths in NYC. Check it out [here!][covid19nyc]

Here's the code, written with D3.js v5:
{% highlight ruby %}
<!DOCTYPE html>
<html>
  <head>
    <meta http-equiv="Content-Type" content="text/html;charset=utf-8">
    <title>COVID-19 in NYC</title>
    <script type="text/javascript" src="https://d3js.org/d3.v5.min.js"></script>
    <style>

        html, body {
            margin: 0;
            padding: 0;
            height: 100%;
            width: 100%;
            font: 10px "Helvetica Neue", Helvetica, Arial, sans-serif;
            position: relative;
        }

        text {
            font-family: arial;
            font-size: 12px;
        }


        path.line {
            fill: none;
            stroke: red;
            stroke-width: 2px;
        }

        .overlay {
                fill: none;
                pointer-events: all;
        }

        .axis path,
        .axis line {
          fill: none;
          stroke: slategray;
          shape-rendering: crispEdges;
        }

        .focus circle {
        fill: none;
        stroke: black;
        }

        .tooltip {
            width: 94px;
            padding: 4px 10px;
            border: 1px solid #aaa;
            border-radius: 4px;
            box-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            position: absolute;
            background-color: white;
            font-size: 14px;
            pointer-events: none;
            -webkit-transition: all 0.25s;
            -moz-transition: all 0.25s;
            -ms-transition: all 0.25s;
            -o-transition: all 0.25s;
            transition: all 0.25s;
        }

        .tooltip div {
            margin: 3px 0;
        }

        .tooltip-date, .tooltip-likes {
            font-weight: bold;
        }

        .grid line {
            stroke: lightgrey;
            stroke-opacity: 0.6;
            shape-rendering: crispEdges
          }
        .grid path {
            stroke-width: 0;

    </style>
  </head>
  <body>

    <script type="text/javascript">

var parseDate = d3.timeParse("%Y-%m-%d"),
    bisectDate = d3.bisector(function(d) { return d.date; }).left,
    formatValue = d3.format(","),
    dateFormatter = d3.timeFormat("%m/%d/%y");

var margin = {left: 50, right: 20, top: 20, bottom: 50 };

var width = 960 - margin.left - margin.right;
var height = 500 - margin.top - margin.bottom;


var max = 0;

var xNudge = 50;
var yNudge = 20;

var tooltip = d3.select("body").append("div")
        .attr("class", "tooltip")
        .style("display", "none");

//Function for converting CSV values from strings to Dates and numbers
var rowConverter = function(d) {
    return {
        date: parseDate(d.date),
        county: d.county,
        cases: parseFloat(d.cases),
        deaths: parseFloat(d.deaths)
    };
}

d3.dsv(",", "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv", rowConverter).then(function(data) {

        dataset = data.filter(function(d) {
              return d["county"] === "New York City"
            });

        max = d3.max(dataset, function(d) { return d.cases; });
        minDate = d3.min(dataset, function(d) {return d.date; });
        maxDate = d3.max(dataset, function(d) { return d.date; });


        var y = d3.scaleLinear()
                    .domain([1,max])
                    .range([height,0]);

        var x = d3.scaleTime()
                    .domain([minDate,maxDate])
                    .range([0,width]);

        var yAxis = d3.axisLeft(y);

        var xAxis = d3.axisBottom(x);

        var line = d3.line()
            .x(function(d){ return x(d.date); })
            .y(function(d){ return y(d.cases); })

        var line2 = d3.line()
            .x(function(d){ return x(d.date); })
            .y(function(d){ return y(d.deaths); })

        function make_x_gridlines() {
            return d3.axisBottom(x)
                .ticks(8)
            }
        function make_y_gridlines() {
            return d3.axisLeft(y)
                .ticks(5)
            }

        var svg = d3.select("body").append("svg").attr("id","svg").attr("height","100%").attr("width","100%");
        var chartGroup = svg.append("g").attr("class","chartGroup").attr("transform","translate("+xNudge+","+yNudge+")");

        chartGroup.append("path")
            .style("stroke","blue")
            .style("fill","none")
            .style("stroke-width","2px")
            .attr("d",function(d){ return line(dataset); })

        chartGroup.append("path")
            .style("stroke","red")
            .style("fill","none")
            .style("stroke-width","2px")
            .attr("d",function(d){ return line2(dataset); })

        var focus = chartGroup.append("g")
            .attr("class", "focus")
            .style("display", "none");

        focus.append("circle")
            .attr("r", 5);

        focus.append("path")
              .style("stroke", "black")
              .style("stroke-width", "1px")
              .style("opacity", "0");

        var tooltipDate = tooltip.append("div")
            .attr("class", "tooltip-date");

        var tooltipCases = tooltip.append("div");
        tooltipCases.append("span")
            .attr("class", "tooltip-title")
            .text("Cases: ");

        var tooltipDeaths = tooltip.append("div");
        tooltipDeaths.append("span")
            .attr("class", "tooltip-title")
            .text("Deaths: ");

        var tooltipCasesValue = tooltipCases.append("span")
            .attr("class", "tooltip-cases");

        var tooltipDeathsValue = tooltipDeaths.append("span")
            .attr("class", "tooltip-deaths");

        chartGroup.append("rect")
            .attr("class", "overlay")
            .attr("width", width)
            .attr("height", height)
            .on("mouseover", function() { focus.style("display", null); tooltip.style("display", null);  })
            .on("mouseout", function() { focus.style("display", "none"); tooltip.style("display", "none"); })
            .on("mousemove", mousemove);

        function mousemove() {
            var x0 = x.invert(d3.mouse(this)[0]),
                i = bisectDate(dataset, x0, 1),
                d0 = dataset[i - 1],
                d1 = dataset[i],
                d = x0 - d0.date > d1.date - x0 ? d1 : d0;
            focus.attr("transform", "translate(" + x(d.date) + "," + y(d.cases) + ")");
            tooltip.attr("style", "left:" + (x(d.date) + 64) + "px;top:" + y(d.cases) + "px;");
            tooltip.select(".tooltip-date").text(dateFormatter(d.date));
            tooltip.select(".tooltip-cases").text(formatValue(d.cases));
            tooltip.select(".tooltip-deaths").text(formatValue(d.deaths));
        }

        chartGroup.append("g")
            .attr("class","axis x")
            .attr("transform","translate(0,"+height+")")
            .call(xAxis);

        chartGroup.append("g")
            .attr("class","axis y")
            .call(yAxis);

        //Create title
        chartGroup.append("text")
            .attr("x", (width / 2))
            .attr("y", 16)
            .attr("text-anchor", "middle")
            .style("font-size", "16px")
            .text("COVID-19 Confirmed Cases and Deaths in NYC");

        chartGroup.append("g")
            .attr("class","grid")
            .attr("transform","translate(0," + height + ")")
            .style("stroke-dasharray",("3,3"))
            .call(make_x_gridlines()
                .tickSize(-height)
                .tickFormat("")
             )
        chartGroup.append("g")
                .attr("class","grid")
                .style("stroke-dasharray",("3,3"))
                .call(make_y_gridlines()
                    .tickSize(-width)
                    .tickFormat("")
                 )


    });

    </script>
  </body>
</html>

{% endhighlight %}

[jhu-viz]: https://www.arcgis.com/apps/opsdashboard/index.html#/bda7594740fd40299423467b48e9ecf6
[covid19nyc]: https://tonyksong.github.io/covid19nyc.html

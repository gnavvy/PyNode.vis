var SQRT3 = Math.sqrt(3);

// svg sizes and margins
var margin = { top: 30, right: 30, bottom: 30, left: 30 };
var width  = $(window).width() - margin.left - margin.right;
var height = $(window).height() - margin.top - margin.bottom;

// The number of columns and rows of the heatmap
var numColumn = 50, numRow = 50;
    
// The maximum radius the hexagons can have to still fit the screen
var hexRadius = d3.min([width/((numColumn+0.5) * SQRT3), height/((numRow+1/3) * 1.5)]);

// Set the new height and width of the SVG based on the max possible
width = numColumn * hexRadius * SQRT3;
heigth = numRow * 1.5 * hexRadius + 0.5 * hexRadius;

// Set the hexagon radius
var hexbin = d3.hexbin().radius(hexRadius);

// Calculate the center positions of each hexagon    
var points = [];
for (var i = 0; i < numRow; i++) {
    for (var j = 0; j < numColumn; j++) {
        points.push([hexRadius * j * SQRT3, hexRadius * i * 1.5]);
    }
}

// Create SVG element
var svg = d3.select("#vis").append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
    .append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")")
;

// Start drawing the hexagons
svg.append("g").selectAll(".hexagon")
    .data(hexbin(points))
    .enter().append("path")
    .attr("class", "hexagon")
    .attr("d", function (d) {
        return "M" + d.x + "," + d.y + hexbin.hexagon();
    })
    .attr("stroke", function (d,i) {
        return "#aaa";
    })
    .attr("stroke-width", "0.5px")
    .style("fill", function (d,i) {
        return "#eee";
    })
    .on("mouseover", mover)
    .on("mouseout", mout)
;

// Function to call when you mouseover a node
function mover(d) {
  var el = d3.select(this).transition().duration(10).style("fill-opacity", 0.1);
}

// Mouseout function
function mout(d) { 
    var el = d3.select(this).transition().duration(500).style("fill-opacity", 1.0);
};
var App = {
    Models: {},
    Collections: {},
    ViewModels: {},
    Views: {}
};

// ------ MODEL ------ //

// ------ COLLECTION ------ //

// ------ VIEW MODEL ------ //
App.ViewModels.HexMapViewModel = Backbone.Model.extend({
    defaults: {
        margin: { 'top': 30, 'right': 30, 'bottom': 30, 'left': 30 },
        canvas: { 'width': 800, 'height': 800 },
        gridDim: { 'x': 50, 'y': 50 }
    },
    set: function(key, val, options) {
        if (!key) return this;   

        var attrs;
        if (typeof(key) === 'object') {  // set multiple attributes
            attrs = key;
            options = val;
        } else {
            attrs = {};
            attrs[key] = val;
        }

        Backbone.Model.prototype.set.call(this, attrs, options);
    },
    url: ''
});

// ------ VIEW ------ //
App.Views.CanvasLayer = Backbone.View.extend({
    initialize: function(options) {},
    render: function() {
        this.initCanvas();
        return this;
    },
    initCanvas: function() {
        var canvas = this.model.get('canvas');
        var margin = this.model.get('margin');

        this.model.set('_figure', this.figureSize());

        this.layer = {};
        this.layer.container = d3.select(this.el).append('svg:svg')
            .attr('width', canvas.width)
            .attr('height', canvas.height);
        this.layer.figure = this.layer.container.append('svg:g')
            .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');
    },
    figureSize: function() {
        var canvas = this.model.get('canvas'),
            margin = this.model.get('margin');
        return {
            width: canvas.width - margin.left - margin.right,
            height: canvas.height - margin.top - margin.bottom
        }
    }
});

App.Views.CoordinateLayer = App.Views.CanvasLayer.extend({
    initialize: function(options) {},
    render: function() {
        this.initCanvas();
        this.initCoordinate();
        return this;
    },
    initCoordinate: function() {
        var gridDim = this.model.get('gridDim');
        var figure = this.model.get('_figure');

        var xRadius = figure.width / ((gridDim.x+0.5) * Math.sqrt(3));
        var yRadius = figure.height / ((gridDim.y+1/3) * 1.5);
        var hexRadius = d3.min([xRadius, yRadius]);
        var hexbin = d3.hexbin().radius(hexRadius);

        var centroids = [];
        for (var y = 0; y < gridDim.y; ++y) {
            for (var x = 0; x < gridDim.x; ++x) {
                centroids.push([hexRadius*x*1.749, hexRadius*y*1.5]);
            }
        }      

        var data = this.model.get('_data_series');
        this.layer.coordinate = this.layer.figure.append("svg:g")
            .selectAll('.hexagon')
            .data(hexbin(centroids))
            .enter().append("path")
            .attr("class", "hexagon")
            .attr("d", function(d) { return "M" + d.x + "," + d.y + hexbin.hexagon(); })
            .attr("stroke", function(d,i) { return "#AAA"; })
            .attr("stroke-width", "0.5px")
            .style("fill", function(d,i) { return "#EEE"; })
            .style("fill-opacity", function(d,i) {
                var x = i % gridDim.x;
                var y = (i-x) / gridDim.x;
                var opacity = data[y][x];
                return opacity;
            });
        return this;
    },
});

App.Views.DataLayer = App.Views.CoordinateLayer.extend({});
App.Views.AnnotationLayer = App.Views.DataLayer.extend({});
App.Views.ListenerLayer = App.Views.AnnotationLayer.extend({});

App.Views.InteractionLayer = App.Views.ListenerLayer.extend({
    options: {  // use options instead of defaults for View
                // ref: http://stackoverflow.com/a/17202740
    },
    initialize: function(options) {
        this.options = _.defaults(options || {}, this.options);
        this.model.set('_data_series', this.collection);
    },
    render: function() {
        this.initCanvas();
        this.initCoordinate();
        this.bindInteraction();
        return this;
    },
    bindInteraction: function() {
        var mouseover = function() {
            d3.select(this).transition().duration(10)
                .style("fill", function (d,i) { return "#0E0"; })
                .style("fill-opacity", 0.5);
        };
        var mouseout = function() {
            d3.select(this).transition().duration(500)
                .style("fill", function (d,i) { return "#EEE"; })
                .style("fill-opacity", 1.0);
        }
        this.layer.coordinate
            .on("mouseover", mouseover)
            .on("mouseout", mouseout);
        return this;
    },
});
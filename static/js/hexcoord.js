var App = {
    Models: {},
    Collections: {},
    ViewModels: {},
    Views: {}
};

// ------ MODEL ------ //
App.Models.DataItem = Backbone.Model.extend({});

// ------ COLLECTION ------ //
App.Collections.DataSeries = Backbone.Collection.extend({
    model: App.Models.DataItem,
    url: '/getData',
    parse: function(response) {
        return response.values;
    }
});


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
        if (typeof key === 'object') {  // set multiple attributes
            attrs = key;
            options = val;
        } else {
            attrs = {};
            attrs[key] = val;
        }
        // globally
        Backbone.Model.prototype.set.call(this, attrs, options);
    },
    url: ''
});

// ------ VIEW ------ //
App.Views.CanvasLayer = Backbone.View.extend({
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
        return this;
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

App.Views.CoordinateLayer = App.Views.CanvasLayer.extend({});

App.Views.DataLayer = App.Views.CoordinateLayer.extend({
    initData: function() {
        this.hexagon();  // create a hexagon generator
        this.data = {};
        this.data.values = this.collection.toJSON();
        this.data.centroids = this.getCentroids();
        return this;
    },
    draw: function() {
        var hexbin = this.model.get('_hexagon');

        var self = this;
        this.layer.hexagons = this.layer.figure.append("svg:g").selectAll('.hexagon')
            .data(hexbin(this.data.centroids))
            .enter().append("path")
            .attr("class", "hexagon")
            .attr("d", function(d) { return "M" + d.x + "," + d.y + hexbin.hexagon(); })
            .attr("stroke", function(d,i) { return self.getValue(i) > 0 ? "#AAA" : "#FFF"; })
            .attr("stroke-width", "0.2px")
            .style("fill", function(d,i) { return self.getColor(i); });
        return this;
    },
    update: function() {
        console.log('update!');
        return this;
    },
    hexagon: function() {
        var gridDim = this.model.get('gridDim');
        var figureArea = this.model.get('_figure');
        var xRadius = figureArea.width / ((gridDim.x+0.5) * Math.sqrt(3));
        var yRadius = figureArea.height / ((gridDim.y+1/3) * 1.5);
        var radius = d3.min([xRadius, yRadius]);
        var hexagon = d3.hexbin().radius(radius);
        this.model.set('_hexRadius', radius);
        this.model.set('_hexagon', hexagon);
        return this;
    },
    getCentroids: function() {
        var radius = this.model.get('_hexRadius');
        var gridDim = this.model.get('gridDim');

        var centroids = [];  // faster than _.range().map().zip()
        for (var y = 0; y < gridDim.y; ++y) {  
            for (var x = 0; x < gridDim.x; ++x) {
                centroids.push([radius * x * 1.749, radius * y * 1.5]);
            }
        }
        return centroids;
    },
    getValue: function(idx) {
        var gridDim = this.model.get('gridDim');
        var x = idx % gridDim.x;
        var y = (idx-x) / gridDim.x;
        return this.data.values.length > 0 ? this.data.values[y][x] : 1.0;
    },
    getColor: function(idx) {
        var opacity = 255 * (1 - this.getValue(idx));
        var hex = Number(parseInt(opacity, 10)).toString(16);
        return "#" + hex + hex + hex;
    }
});

App.Views.AnnotationLayer = App.Views.DataLayer.extend({});
App.Views.ListenerLayer = App.Views.AnnotationLayer.extend({});

App.Views.InteractionLayer = App.Views.ListenerLayer.extend({
    initialize: function() {
        _.bindAll(this, 'render');
        this.collection.bind('reset', this.render);
        this.collection.fetch({reset: true});
    },
    render: function() {
        return this.initCanvas().initData().draw().bindInteraction();
    },
    bindInteraction: function() {
        var mouseover = function() {
            d3.select(this).transition().duration(10)
                .style("fill", function() { return "#0E0"; })
                .style("fill-opacity", 0.5);
        };
        var mouseout = function() {
            d3.select(this).transition().duration(500)
                .style("fill", function() { return "#EEE"; })
                .style("fill-opacity", 1.0);
        };
        this.layer.hexagons
            .on("mouseover", mouseover)
            .on("mouseout", mouseout);
        return this;
    }
});
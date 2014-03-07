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
        if (!key) {
            return this;
        }

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
        return this;
    },
    url: ''
});

// ------ VIEW ------ //
App.Views.CanvasLayer = Backbone.View.extend({
    initCanvas: function() {
        if (_(this).has('layer') && _(this.layer).has('figure')) {
            return this;
        }

        var canvas = this.model.get('canvas');
        var margin = this.model.get('margin');

        this.model.set('_figure', {
            width: canvas.width - margin.left - margin.right,
            height: canvas.height - margin.top - margin.bottom
        });

        this.layer = {};
        this.layer.container = d3.select(this.el).append('svg:svg')
            .style('width', canvas.width)
            .style('height', canvas.height);
        this.layer.figure = this.layer.container.append('svg:g')
            .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');
        return this;
    }
});

App.Views.CoordinateLayer = App.Views.CanvasLayer.extend({
    initCoordinate: function() {
        this.createHexagonSkeleton();
        var hexbin = this.model.get('_hexagon');
        var centroids = this.calculateCentroids();
        this.layer.hexagons = this.layer.figure.append("svg:g").selectAll('.hexagon')
            .data(hexbin(centroids))
            .enter().append("path")
            .attr("d", function(d) { return "M" + d.x + "," + d.y + hexbin.hexagon(); })
            .style("class", "hexagon")
            .style("stroke-width", "0.2px")
            .style("stroke", "#AAA")
            .style("fill", "#FFF");
        return this;
    },
    createHexagonSkeleton: function() {
        var gridDim = this.model.get('gridDim');
        var figureArea = this.model.get('_figure');
        var xRadius = figureArea.width / ((gridDim.x+0.5) * Math.sqrt(3));
        var yRadius = figureArea.height / ((gridDim.y+1/3) * 1.5);
        var radius = d3.min([xRadius, yRadius]);
        var hexagon = d3.hexbin().radius(radius);
        this.model.set({ '_hexRadius': radius, '_hexagon': hexagon});
    },
    calculateCentroids: function() {
        var radius = this.model.get('_hexRadius');
        var gridDim = this.model.get('gridDim');
        var centroids = [];  // faster than _.range().map().zip()
        for (var y = 0; y < gridDim.y; ++y) {
            for (var x = 0; x < gridDim.x; ++x) {
                centroids.push([radius * x * 1.749, radius * y * 1.5]);
            }
        }
        return centroids;
    }
});

App.Views.DataLayer = App.Views.CoordinateLayer.extend({
    updateData: function() {
        this.data = this.collection.toJSON();
        return this;
    },
    drawData: function() {  // update
        // first check if hexmap coordinate is created
        if (!_(this).has('layer') || !_(this.layer).has('hexagons')) {
            this.initCoordinate();
        }
        var self = this;
        this.layer.hexagons
            .style("stroke", function(d,i) { return self.getValue(i) > 0 ? "#AAA" : "#FFF"; })
            .style("fill", function(d,i) { return self.getColor(i); });
        return this;
    },
    getValue: function(idx) {
        var gridDim = this.model.get('gridDim');
        var x = idx % gridDim.x;
        var y = (idx-x) / gridDim.x;
        return this.data.length > 0 ? this.data[y][x] : 0.0;
    },
    getColor: function(idx) {
        var value = this.getValue(idx);
        return this.model.get('colorScheme')(value);
    }
});

App.Views.AnnotationLayer = App.Views.DataLayer.extend({});

App.Views.InteractionLayer = App.Views.AnnotationLayer.extend({
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
        var click = function() {
            d3.select(this).transition().duration(10)
                .style("fill", function() { return "#F00"; })
                .style("fill-opacity", 0.5);
        };
        this.layer.hexagons
            .on("mouseover", mouseover)
            .on("mouseout", mouseout)
            .on("click", click);
        return this;
    }
});

App.Views.HexMapView = App.Views.InteractionLayer.extend({
    initialize: function() {
        _(this).bindAll('render');
        this.collection.bind('reset', this.render);
        this.collection.fetch({reset: true});

        var color = d3.scale.linear().domain([0.0, 1.0, 100])
            .range(['orange', 'white']);
        this.model.set('colorScheme', color);
    },
    render: function() {
        return this.initCanvas().updateData().drawData().bindInteraction();
    }
});

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
        return response.data;
    }
});

// ------ VIEW MODEL ------ //
App.ViewModels.HexMapViewModel = Backbone.Model.extend({
    defaults: {
        margin: { 'top': 30, 'right': 30, 'bottom': 30, 'left': 30 },
        canvas: { 'width': 800, 'height': 800 },
        gridDim: { 'x': 50, 'y': 50 }
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

        this.param.domain = {
            width: canvas.width - margin.left - margin.right,
            height: canvas.height - margin.top - margin.bottom
        };

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
        var self = this;
        this.layer.hexagons = this.layer.figure.append("svg:g").selectAll('.hexagon')
            .data(this.param.hexagon(this.param.centroids))
            .enter().append("path")
            .attr("d", function(d) {
                return "M" + d.x + "," + d.y + self.param.hexagon.hexagon();
            })
            .style("class", "hexagon")
            .style("stroke-width", "0.2px")
            .style("stroke", "#AAA")
            .style("fill", "#FFF");
        return this;
    },
    createHexagonSkeleton: function() {
        var gridDim = this.model.get('gridDim');
        var xRadius = this.param.domain.width / ((gridDim.x+0.5) * Math.sqrt(3));
        var yRadius = this.param.domain.height / ((gridDim.y+1/3) * 1.5);
        var radius = d3.min([xRadius, yRadius]);
        var centroids = [];  // faster than _.range().map().zip()
        for (var y = 0; y < gridDim.y; ++y) {
            for (var x = 0; x < gridDim.x; ++x) {
                centroids.push([radius * x * 1.749, radius * y * 1.5]);
            }
        }
        this.param.hexRadius = radius;
        this.param.centroids = centroids;
        this.param.hexagon = d3.hexbin().radius(radius);
    }
});

App.Views.DataLayer = App.Views.CoordinateLayer.extend({
    updateData: function() {
        this.data = this.collection.toJSON();
        if (this.data.length > 0) {
            this.data.values = this.collection.toJSON()[0]['values'];
            this.data.indices = this.collection.toJSON()[0]['indices'];
            this.data.selected = this.collection.toJSON()[0]['selected'];
        }
        return this;
    },
    drawData: function() {  // update
        // first check if hexmap coordinate is created
        if (!_(this).has('layer') || !_(this.layer).has('hexagons')) {
            this.initCoordinate();
        }
        var self = this;
        this.layer.hexagons.style("fill", function(d,i) {
            return self.getColor(i);
        });
        return this;
    },
    getValue: function(idx) {
        if (this.data.values === undefined) {
            return 0.5;  // set to white by default
        }
        var gridDim = this.model.get('gridDim');
        var x = idx % gridDim.x;
        var y = (idx-x) / gridDim.x;
        return this.data.values[y][x];
    },
    getColor: function(idx) {
        var value = this.getValue(idx);
        return this.param.colorScheme(value);
    }
});

App.Views.AnnotationLayer = App.Views.DataLayer.extend({
    annotate: function() {
        if (!this.data.indices || !this.param.centroids) {
            return this;
        }

        var landmarkCentroids = [];
        for (var i = 0; i < this.data.indices.length; ++i) {
            var idx = this.data.indices[i];
            if (this.param.centroids[idx]) {
                landmarkCentroids.push(this.param.centroids[idx]);
            }
        }

        var self = this;
        this.layer.landmarks = this.layer.figure.append("svg:g").selectAll('.hexagon')
            .data(this.param.hexagon(landmarkCentroids))
            .enter().append("path")
            .attr("d", function(d) {
                return "M" + d.x + "," + d.y + self.param.hexagon.hexagon();
            })
            .style("class", "hexagon")
            .style("stroke-width", "2px")
            .style("stroke", function(d, i) {
                return self.isSelected(i) ? "F00" : "000";
            })
            .style("fill", "transparent")
        ;
        return this;
    },
    isLandmark: function(idx) {
        return this.data.indices && _(this.data.indices).contains(idx)
    },
    isSelected: function(idx) {
        return this.data.selected && this.data.selected === idx
    }
});

App.Views.InteractionLayer = App.Views.AnnotationLayer.extend({
    bindInteraction: function() {
        var mouseover = function() {
            d3.select(this).transition().duration(10)
                .style("fill-opacity", 0.1);
        };
        var mouseout = function() {
            d3.select(this).transition().duration(500)
                .style("fill-opacity", 1.0);
        };
        var click = function() {
            d3.select(this).transition().duration(10)
                .style("fill", function() { return "#F00"; })
                .style("fill-opacity", 0.5);
        };
        this.layer.landmarks
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

        this.param = {};
        this.param.colorScheme = d3.scale.linear().domain([0.0, 0.5, 1.0])
            .range(["red", 'white', 'green']);
    },
    render: function() {
        return this.initCanvas()
            .updateData().drawData()
            .annotate()
            .bindInteraction();
    }
});

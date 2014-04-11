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
    url: '/'
    // url: '/getData',
    // parse: function(response) {
    //     return response.data;
    // }
});

// ------ VIEW MODEL ------ //
App.ViewModels.HexMapViewModel = Backbone.Model.extend({
    defaults: {
        // margin: { 'top': 30, 'right': 30, 'bottom': 30, 'left': 30 },
        // canvas: { 'width': 800, 'height': 800 },
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

        if (!_(this).has("p")) {
            this.p = {};
        }

        var w = window.innerWidth;
        var h = window.innerHeight;
        this.p.domain = { width: w, height: h };

        this.p.canvas = document.createElement("div");
        document.body.appendChild(this.p.canvas);

        this.p.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.p.renderer.setSize(w, h, undefined);
        // this.p.renderer.domElement.addEventListener('mousemove', onDocumentMouseMove, false);
        // this.p.renderer.domElement.addEventListener('mousedown', onDocumentMouseDown, false);
        // this.p.renderer.domElement.addEventListener('mouseup', onDocumentMouseUp, false);
        this.p.canvas.appendChild(this.p.renderer.domElement);

        this.p.scene = new THREE.Scene();
        this.p.camera = new THREE.PerspectiveCamera(45, w / h, 1, 100000);
        this.p.camera.position.set(6000, 3000, 6000);
        this.p.camera.lookAt(new THREE.Vector3(0, 0, 0));
        this.p.scene.add(this.p.camera);

        return this;
    }
});

App.Views.CoordinateLayer = App.Views.CanvasLayer.extend({
    initCoordinate: function() {
        this.p.scene.add(new THREE.AxisHelper(3200));
        var planeMesh = new THREE.Mesh(
            new THREE.PlaneGeometry(3200, 3200, 20, 20),
            new THREE.MeshBasicMaterial({ color: 0xe0e0e0, wireframe: true })
        );

        var xy_plane = planeMesh.clone();
        var xz_plane = planeMesh.clone();
        var yz_plane = planeMesh.clone();

        xy_plane.position.set(1600, 1600, 0);
        xz_plane.rotation.x = Math.PI/2;
        xz_plane.position.set(1600, 0, 1600);
        yz_plane.rotation.y = Math.PI/2;
        yz_plane.position.set(0, 1600, 1600);

        this.p.scene.add(xy_plane);
        this.p.scene.add(xz_plane);
        this.p.scene.add(yz_plane);

        return this;
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

        var self = this;

        var landmarkCentroids = [];
        for (var i = 0; i < this.data.indices.length; ++i) {
            var idx = this.data.indices[i];
            if (this.param.centroids[idx]) {
                landmarkCentroids.push(this.param.centroids[idx]);
            }
        }
        this.layer.landmarks = this.layer.figure.append("svg:g").selectAll('.landmarks')
            .data(this.param.hexagon(landmarkCentroids))
            .enter().append("path")
            .attr("d", function(d) {
                return "M" + d.x + "," + d.y + self.param.hexagon.hexagon();
            })
            .style("class", "hexagon")
            .style("stroke-width", "1.5px")
            .style("stroke", "#333")
            .style("fill", "transparent");

//        var temp = [this.param.centroids[this.data.selected]];
//        this.layer.selected = this.layer.figure.append("svg:g").selectAll(".selected")
//            .data(this.param.hexagon(temp))
//            .enter().append("path")
//            .attr("d", function(d) {
//                return "M" + d.x + "," + d.y + self.param.hexagon.hexagon();
//            })
//            .style("class", "hexagon")
//            .style("stroke-width", "2px")
//            .style("stroke", "F00")
//            .style("fill", "transparent");

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
        // var mouseover = function() {
        //     d3.select(this).transition().duration(10)
        //         .style("fill-opacity", 0.1);
        // };
        // var mouseout = function() {
        //     d3.select(this).transition().duration(500)
        //         .style("fill-opacity", 1.0);
        // };
        // var click = function() {
        //     d3.select(this).transition().duration(10)
        //         .style("fill", function() { return "#F00"; })
        //         .style("fill-opacity", 0.5);
        // };
        // this.layer.landmarks
        //     .on("mouseover", mouseover)
        //     .on("mouseout", mouseout)
        //     .on("click", click);
        // return this;
        window.addEventListener('resize', onWindowResize, false);
    }
});

// App.Views.HexMapView = App.Views.InteractionLayer.extend({
App.Views.HexMapView = App.Views.CoordinateLayer.extend({
    initialize: function() {
        _(this).bindAll('render');
        this.collection.bind('reset', this.render);
        this.collection.fetch({reset: true});

        // this.param = {};
        // this.param.colorScheme = d3.scale.linear().domain([0.0, 0.5, 1.0])
        //     .range(["red", 'white', 'green']);
        this.initCanvas().initCoordinate();
    },
    render: function() {
        this.p.renderer.render(this.p.scene, this.p.camera);
        // return this.initCanvas().initCoordinate();
            // .updateData().drawData()
            // .annotate()
            // .bindInteraction();
    },
    animate: function() {
        requestAnimationFrame(this.animate);
        this.render();
    }
});
